"""
钉钉多维表格通知模块 (DingTalk AI Table Notifier)
=================================================
基于 MCP 协议通过 DingTalkAITableClient 向钉钉多维表格写入交易信号与运行记录。

特性:
1. 动态字段解析：调用 get_tables() 按中文列名动态解析字段 ID，不硬编码 fld_xxx。
2. 9 个标准字段：日期 / 股票代码 / 股票名称 / 操作 / 触发价 / 止损价 / 止盈价 / ML置信度 / 触发原因。
3. 异步兼容：内部使用 async with 连接 MCP，对外提供同步 notify_trade() / notify_signals() 接口。
4. 容错防崩：未配置 DINGTALK_MCP_URL 时安全跳过并打印预览。
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from .notifier import BaseNotifier
from data.fetcher import STOCK_NAMES
from utils.logger import get_logger

log = get_logger("xlt.dingtalk")


# 标准中文列名定义
STANDARD_COLUMNS = [
    "日期",
    "股票代码",
    "股票名称",
    "操作",
    "触发价",
    "止损价",
    "止盈价",
    "ML置信度",
    "触发原因",
]


def _default_client_factory(server_url: str):
    """默认客户端工厂"""
    from .vendor.dingtalk_client import DingTalkAITableClient
    return DingTalkAITableClient(server_url=server_url)


class DingTalkNotifier(BaseNotifier):
    """
    钉钉多维表格通知器
    """

    def __init__(
        self,
        enabled: bool = False,
        mcp_url: str = "",
        base_id: str = "",
        table_id: str = "",
        client_factory: Optional[Callable] = None,
    ):
        self.enabled = enabled
        # 优先使用显式传入的 mcp_url，若为空则从环境变量 DINGTALK_MCP_URL 读取
        self.mcp_url = mcp_url or os.environ.get("DINGTALK_MCP_URL", "")
        self.base_id = base_id
        self.table_id = table_id
        self.client_factory = client_factory or _default_client_factory
        self._field_id_cache: Dict[str, str] = {}

    def is_configured(self) -> bool:
        """检查必要配置是否完整"""
        return bool(self.enabled and self.mcp_url and self.base_id and self.table_id)

    def resolve_field_ids(self, tables_meta: Any, target_table_id: Optional[str] = None) -> Dict[str, str]:
        """
        从 get_tables() 返回的元数据中动态解析 {中文列名: fieldId} 映射。
        支持多种 MCP 返回结构（包含 tables 列表、单个 table 对象或直接 field 列表）。
        """
        tid = target_table_id or self.table_id
        fields = []

        if isinstance(tables_meta, dict):
            # 情况 1: {"tables": [{"id": "...", "fields": [...]}]}
            if "tables" in tables_meta and isinstance(tables_meta["tables"], list):
                for t in tables_meta["tables"]:
                    if t.get("id") == tid or t.get("tableId") == tid or len(tables_meta["tables"]) == 1:
                        fields = t.get("fields") or t.get("fieldList") or []
                        break
            # 情况 2: {"fields": [...]} 或 {"fieldList": [...]}
            elif "fields" in tables_meta:
                fields = tables_meta["fields"]
            elif "fieldList" in tables_meta:
                fields = tables_meta["fieldList"]
        elif isinstance(tables_meta, list):
            for t in tables_meta:
                if isinstance(t, dict) and (t.get("id") == tid or t.get("tableId") == tid or len(tables_meta) == 1):
                    fields = t.get("fields") or t.get("fieldList") or []
                    break

        mapping = {}
        for f in fields:
            if isinstance(f, dict):
                fid = f.get("id") or f.get("fieldId")
                fname = f.get("name") or f.get("fieldName")
                if fid and fname:
                    mapping[str(fname).strip()] = str(fid).strip()

        return mapping

    def build_record_cells(
        self,
        field_mapping: Dict[str, str],
        signal_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        根据中文列名映射组装钉钉多维表格单行 cells 字典。
        """
        date_val = str(signal_data.get("date") or datetime.now().strftime("%Y-%m-%d"))
        symbol_val = str(signal_data.get("symbol", "")).zfill(6)
        name_val = str(signal_data.get("name") or STOCK_NAMES.get(symbol_val, symbol_val))
        action_val = str(signal_data.get("action", ""))
        price_val = float(signal_data.get("price", 0.0))
        sl_val = float(signal_data.get("stop_loss_price", 0.0))
        tp_val = float(signal_data.get("take_profit_price", 0.0))
        ml_val = float(signal_data.get("ml_confidence", 0.0))
        reason_val = str(signal_data.get("reason", ""))

        raw_map = {
            "日期": date_val,
            "股票代码": symbol_val,
            "股票名称": name_val,
            "操作": action_val,
            "触发价": round(price_val, 2),
            "止损价": round(sl_val, 2),
            "止盈价": round(tp_val, 2),
            "ML置信度": round(ml_val, 3),
            "触发原因": reason_val,
        }

        cells = {}
        for col_name, val in raw_map.items():
            if col_name in field_mapping:
                fld_id = field_mapping[col_name]
                cells[fld_id] = val

        return cells

    async def async_create_signal_records(self, signals: List[Dict[str, Any]]) -> bool:
        """异步写入信号记录到钉钉多维表格"""
        if not signals:
            return True

        if not self.is_configured():
            log.info("[钉钉通知] 未配置完整 DINGTALK_MCP_URL / base_id / table_id，跳过网络写入。")
            log.info(f"[钉钉通知待写入预览]: 共 {len(signals)} 条信号")
            for s in signals:
                log.info(f"  • {s.get('symbol')} {s.get('name')} | {s.get('action')} | 现价:{s.get('price')} | 止损:{s.get('stop_loss_price')} | 原因:{s.get('reason')}")
            return False

        try:
            async with self.client_factory(self.mcp_url) as client:
                # 1. 动态获取表结构并解析中文列名 ID
                if not self._field_id_cache:
                    log.info(f"[钉钉通知] 正在从 Base {self.base_id} 动态读取 Table {self.table_id} 字段结构...")
                    tables_meta = await client.get_tables(self.base_id, [self.table_id])
                    self._field_id_cache = self.resolve_field_ids(tables_meta, self.table_id)
                    log.info(f"[钉钉通知] 字段解析完成: {self._field_id_cache}")

                # 2. 组装每行 cells
                records_to_create = []
                for sig in signals:
                    cells = self.build_record_cells(self._field_id_cache, sig)
                    if cells:
                        records_to_create.append({"cells": cells})

                if not records_to_create:
                    log.warning("[钉钉通知] 未能匹配到任何有效字段，写入终止")
                    return False

                # 3. 调用 create_records 批量写入
                log.info(f"[钉钉通知] 正在向钉钉多维表格写入 {len(records_to_create)} 条记录...")
                res = await client.create_records(self.base_id, self.table_id, records_to_create)
                log.info(f"[钉钉通知] 写入成功: {res}")
                return True
        except Exception as e:
            log.error(f"[钉钉通知] 写入异常: {e}")
            return False

    def notify_trade(
        self,
        action: str,
        symbol: str,
        price: float,
        size: int,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        同步兼容接口：将单笔交易信号写入钉钉多维表格
        """
        sig_data = {
            "date": (extra.get("date") if extra else None) or datetime.now().strftime("%Y-%m-%d"),
            "symbol": symbol,
            "name": (extra.get("name") if extra else None) or STOCK_NAMES.get(symbol, symbol),
            "action": action,
            "price": price,
            "stop_loss_price": extra.get("stop_loss_price", 0.0) if extra else 0.0,
            "take_profit_price": extra.get("take_profit_price", 0.0) if extra else 0.0,
            "ml_confidence": extra.get("ml_confidence", 0.0) if extra else 0.0,
            "reason": reason,
        }
        return self.notify_signals([sig_data])

    def notify_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """
        同步兼容接口：批量将信号列表写入钉钉多维表格
        """
        if not signals:
            return True
        try:
            return asyncio.run(self.async_create_signal_records(signals))
        except RuntimeError:
            # 若当前线程已有正在运行的 event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.async_create_signal_records(signals))

    def notify_report(self, report: str) -> bool:
        """每日文本报告通知（打日志或保留占位）"""
        if not self.is_configured():
            return False
        log.info(f"[钉钉通知] 报告文本推送: {report[:100]}...")
        return True

    def notify_error(self, error: str) -> bool:
        """异常告警通知"""
        log.error(f"[钉钉通知] 系统异常: {error}")
        return False

    def notify_kill_switch(self) -> bool:
        """紧急停止通知"""
        log.critical("[钉钉通知] Kill Switch 已触发！")
        return False
