"""
风控模块 — Kill Switch + 每日亏损限制 + 仓位检查
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime

from .logger import log
from .strategy import Signal, Action


class RiskManager:
    """
    风控管理器

    - Kill Switch: 创建 .kill_switch 文件即停止所有交易
    - 每日最大亏损限制
    - 仓位上限检查
    - 信号合法性校验
    """

    def __init__(
        self,
        kill_switch_file: str = ".kill_switch",
        max_daily_loss_pct: float = 0.03,
        max_single_pct: float = 0.20,
        max_total_pct: float = 0.60,
    ):
        self.kill_switch_file = Path(kill_switch_file)
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_single_pct = max_single_pct
        self.max_total_pct = max_total_pct

        self._daily_start_value: float | None = None
        self._stopped = False
        self._stop_event = threading.Event()

        # 启动 Kill Switch 文件监控线程
        self._watcher_thread = threading.Thread(
            target=self._watch_kill_switch, daemon=True
        )
        self._watcher_thread.start()

    # ──────────────────────────────────────
    # Kill Switch
    # ──────────────────────────────────────
    def activate_kill_switch(self):
        """手动激活 Kill Switch"""
        self.kill_switch_file.write_text(
            f"activated at {datetime.now().isoformat()}"
        )
        self._stopped = True
        log.critical("[风控] Kill Switch 已手动激活！")

    def deactivate_kill_switch(self):
        """解除 Kill Switch"""
        if self.kill_switch_file.exists():
            self.kill_switch_file.unlink()
        self._stopped = False
        log.info("[风控] Kill Switch 已解除")

    def is_stopped(self) -> bool:
        """检查是否已停止"""
        return self._stopped or self.kill_switch_file.exists()

    def _watch_kill_switch(self):
        """后台线程：监控 Kill Switch 文件"""
        while not self._stop_event.is_set():
            if self.kill_switch_file.exists() and not self._stopped:
                self._stopped = True
                log.critical(
                    "[风控] 检测到 Kill Switch 文件！所有交易已停止。"
                    f"删除 {self.kill_switch_file} 解除"
                )
            self._stop_event.wait(0.5)  # 每 0.5s 检查一次

    def stop(self):
        """停止监控线程"""
        self._stop_event.set()

    # ──────────────────────────────────────
    # 每日亏损限制
    # ──────────────────────────────────────
    def set_daily_start_value(self, value: float):
        """设置今日开盘净值（每日调用一次）"""
        self._daily_start_value = value

    def check_daily_loss(self, current_value: float) -> bool:
        """检查今日亏损是否超限，超限返回 True"""
        if self._daily_start_value is None:
            return False
        loss_pct = (self._daily_start_value - current_value) / self._daily_start_value
        if loss_pct >= self.max_daily_loss_pct:
            log.warning(
                f"[风控] 今日亏损 {loss_pct*100:.2f}% >= {self.max_daily_loss_pct*100:.0f}%，"
                "触发暂停"
            )
            return True
        return False

    # ──────────────────────────────────────
    # 信号校验
    # ──────────────────────────────────────
    def validate_signal(
        self,
        signal: Signal,
        current_cash: float,
        current_position_value: float,
        total_value: float,
        current_position_size: int,
    ) -> tuple[bool, str]:
        """
        校验信号是否可执行

        Returns:
            (is_valid, reason)
        """
        # Kill Switch
        if self.is_stopped():
            return False, "Kill Switch 已激活，拒绝交易"

        if signal.action == Action.HOLD:
            return False, "HOLD 信号，跳过"

        if signal.action == Action.BUY:
            # 总仓位限制
            if current_position_value >= total_value * self.max_total_pct:
                return False, f"总仓位已达 {self.max_total_pct*100:.0f}% 上限"

            # 单股仓位限制
            max_amount = total_value * self.max_single_pct
            if signal.price > 0:
                max_size = int(max_amount / signal.price / 100) * 100
                if max_size < 100:
                    return False, "单股仓位计算不足100股"

            # 资金检查
            if current_cash < signal.price * 100:
                return False, "可用资金不足买100股"

            return True, "买入校验通过"

        if signal.action == Action.SELL:
            if current_position_size <= 0:
                return False, "无持仓，无法卖出"
            return True, "卖出校验通过"

        return False, "未知信号类型"

    # ──────────────────────────────────────
    # 状态报告
    # ──────────────────────────────────────
    def status(self) -> dict:
        return {
            "kill_switch_active": self.is_stopped(),
            "daily_start_value": self._daily_start_value,
            "max_daily_loss": f"{self.max_daily_loss_pct*100:.0f}%",
            "max_single_position": f"{self.max_single_pct*100:.0f}%",
            "max_total_position": f"{self.max_total_pct*100:.0f}%",
        }
