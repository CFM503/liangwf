"""
Unit tests for DingTalk AI Table Notifier with Mock Client.
Verifies dynamic Chinese column name resolution, 9-field cell assembly,
and unconfigured safety fallback without real credentials.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project roots
sys.path.insert(0, str(Path(__file__).parent.parent / "XiaoLiangTrader"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.dingtalk_notifier import DingTalkNotifier, STANDARD_COLUMNS
from bot.notifier import BaseNotifier


class MockDingTalkClient:
    """Mock DingTalkAITableClient for testing MCP interaction"""
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.created_records = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get_tables(self, base_id: str, table_ids: list):
        return {
            "tables": [
                {
                    "id": "tbl_test_001",
                    "name": "短线交易信号看板",
                    "fields": [
                        {"id": "fld_date", "name": "日期", "type": "text"},
                        {"id": "fld_sym", "name": "股票代码", "type": "text"},
                        {"id": "fld_name", "name": "股票名称", "type": "text"},
                        {"id": "fld_act", "name": "操作", "type": "text"},
                        {"id": "fld_price", "name": "触发价", "type": "number"},
                        {"id": "fld_sl", "name": "止损价", "type": "number"},
                        {"id": "fld_tp", "name": "止盈价", "type": "number"},
                        {"id": "fld_conf", "name": "ML置信度", "type": "number"},
                        {"id": "fld_rsn", "name": "触发原因", "type": "text"},
                    ]
                }
            ]
        }

    async def create_records(self, base_id: str, table_id: str, records: list):
        self.created_records.extend(records)
        return {"code": 0, "msg": "success", "createdCount": len(records)}


class TestDingTalkNotifier(unittest.TestCase):
    def test_base_notifier_subclass(self):
        """测试 DingTalkNotifier 继承 BaseNotifier"""
        self.assertTrue(issubclass(DingTalkNotifier, BaseNotifier))

    def test_field_resolution(self):
        """测试动态从元数据解析9个标准中文列名"""
        notifier = DingTalkNotifier(base_id="base_123", table_id="tbl_test_001")
        mock_meta = {
            "tables": [
                {
                    "id": "tbl_test_001",
                    "fields": [
                        {"id": "fld_01", "name": "日期"},
                        {"id": "fld_02", "name": "股票代码"},
                        {"id": "fld_03", "name": "股票名称"},
                        {"id": "fld_04", "name": "操作"},
                        {"id": "fld_05", "name": "触发价"},
                        {"id": "fld_06", "name": "止损价"},
                        {"id": "fld_07", "name": "止盈价"},
                        {"id": "fld_08", "name": "ML置信度"},
                        {"id": "fld_09", "name": "触发原因"},
                    ]
                }
            ]
        }
        mapping = notifier.resolve_field_ids(mock_meta)
        for col in STANDARD_COLUMNS:
            self.assertIn(col, mapping)
        self.assertEqual(mapping["日期"], "fld_01")
        self.assertEqual(mapping["股票代码"], "fld_02")
        self.assertEqual(mapping["ML置信度"], "fld_08")

    def test_build_record_cells(self):
        """测试根据字段映射组装每行 cells"""
        notifier = DingTalkNotifier(base_id="base_123", table_id="tbl_test_001")
        field_mapping = {
            "日期": "fld_date",
            "股票代码": "fld_sym",
            "股票名称": "fld_name",
            "操作": "fld_act",
            "触发价": "fld_price",
            "止损价": "fld_sl",
            "止盈价": "fld_tp",
            "ML置信度": "fld_conf",
            "触发原因": "fld_rsn",
        }
        sig = {
            "date": "2026-08-14",
            "symbol": "600150",
            "name": "中国船舶",
            "action": "BUY",
            "price": 41.16,
            "stop_loss_price": 37.86,
            "take_profit_price": 47.33,
            "ml_confidence": 0.582,
            "reason": "MA5/20金叉放量",
        }
        cells = notifier.build_record_cells(field_mapping, sig)
        self.assertEqual(cells["fld_date"], "2026-08-14")
        self.assertEqual(cells["fld_sym"], "600150")
        self.assertEqual(cells["fld_name"], "中国船舶")
        self.assertEqual(cells["fld_act"], "BUY")
        self.assertEqual(cells["fld_price"], 41.16)
        self.assertEqual(cells["fld_sl"], 37.86)
        self.assertEqual(cells["fld_tp"], 47.33)
        self.assertEqual(cells["fld_conf"], 0.582)
        self.assertEqual(cells["fld_rsn"], "MA5/20金叉放量")

    def test_mock_client_send_signals(self):
        """测试使用 MockDingTalkClient 验证完整的写入流程"""
        mock_client = MockDingTalkClient("https://mock-mcp-url")
        notifier = DingTalkNotifier(
            enabled=True,
            mcp_url="https://mock-mcp-url",
            base_id="base_mock_123",
            table_id="tbl_test_001",
            client_factory=lambda url: mock_client,
        )

        test_signals = [
            {
                "date": "2026-08-14",
                "symbol": "600519",
                "name": "贵州茅台",
                "action": "BUY",
                "price": 1450.0,
                "stop_loss_price": 1334.0,
                "take_profit_price": 1667.5,
                "ml_confidence": 0.65,
                "reason": "MA金叉放量",
            }
        ]

        ok = notifier.notify_signals(test_signals)
        self.assertTrue(ok)
        self.assertEqual(len(mock_client.created_records), 1)
        created_cells = mock_client.created_records[0]["cells"]
        self.assertEqual(created_cells["fld_sym"], "600519")
        self.assertEqual(created_cells["fld_price"], 1450.0)

    def test_unconfigured_safety_fallback(self):
        """测试未配置 MCP URL 时安全跳过不报错"""
        notifier = DingTalkNotifier(
            enabled=False,
            mcp_url="",
            base_id="",
            table_id="",
        )
        self.assertFalse(notifier.is_configured())
        # 即使调用 notify_signals 也不应抛出异常，安全返回 False
        res = notifier.notify_signals([{"symbol": "600519", "action": "BUY", "price": 1500.0}])
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
