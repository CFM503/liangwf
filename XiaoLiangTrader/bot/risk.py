"""
风控模块 — 这是保命的东西
============================
1. Kill Switch：创建文件即停止交易
2. 每日亏损限制
3. 涨跌停检查
4. 仓位上限
"""

import threading
from pathlib import Path
from datetime import datetime

from strategy.signals import Signal, Action
from utils.logger import get_logger

log = get_logger("xlt.risk")


class RiskManager:
    def __init__(
        self,
        kill_switch_file: str = ".kill_switch",
        max_daily_loss_pct: float = 0.03,
        max_single_pct: float = 0.20,
        max_total_pct: float = 0.60,
        limit_up: float = 0.095,
        limit_down: float = -0.095,
    ):
        self.kill_switch_file = Path(kill_switch_file)
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_single_pct = max_single_pct
        self.max_total_pct = max_total_pct
        self.limit_up = limit_up
        self.limit_down = limit_down

        self._daily_start_value: float | None = None
        self._stopped = False
        self._stop_event = threading.Event()

        # 后台监控 Kill Switch 文件
        threading.Thread(target=self._watch_kill_switch, daemon=True).start()

    # ── Kill Switch ──
    def activate_kill_switch(self):
        self.kill_switch_file.write_text(f"activated {datetime.now().isoformat()}")
        self._stopped = True
        log.critical("[风控] Kill Switch 已激活！")

    def deactivate_kill_switch(self):
        if self.kill_switch_file.exists():
            self.kill_switch_file.unlink()
        self._stopped = False
        log.info("[风控] Kill Switch 已解除")

    def is_stopped(self) -> bool:
        return self._stopped or self.kill_switch_file.exists()

    def _watch_kill_switch(self):
        while not self._stop_event.is_set():
            if self.kill_switch_file.exists() and not self._stopped:
                self._stopped = True
                log.critical("[风控] 检测到 Kill Switch 文件！")
            self._stop_event.wait(0.5)

    def stop(self):
        self._stop_event.set()

    # ── 每日亏损 ──
    def set_daily_start_value(self, value: float):
        self._daily_start_value = value

    def check_daily_loss(self, current_value: float) -> bool:
        if self._daily_start_value is None:
            return False
        loss = (self._daily_start_value - current_value) / self._daily_start_value
        if loss >= self.max_daily_loss_pct:
            log.warning(f"[风控] 今日亏损 {loss*100:.2f}% 超限！")
            return True
        return False

    # ── 涨跌停检查 ──
    def is_limit_up(self, pct_change: float) -> bool:
        """是否涨停（涨停不追）"""
        return pct_change >= self.limit_up

    def is_limit_down(self, pct_change: float) -> bool:
        """是否跌停（跌停不卖，大概率卖不出去）"""
        return pct_change <= self.limit_down

    # ── 信号校验 ──
    def validate_signal(
        self, signal: Signal, cash: float, pos_value: float,
        total_value: float, pos_size: int, pct_change: float = 0.0,
    ) -> tuple[bool, str]:
        if self.is_stopped():
            return False, "Kill Switch 激活"

        if signal.action == Action.HOLD:
            return False, "HOLD"

        if signal.action == Action.BUY:
            # 涨停不追
            if self.is_limit_up(pct_change):
                return False, f"涨停不追 (涨{pct_change*100:.1f}%)"
            # 仓位限制
            if pos_value >= total_value * self.max_total_pct:
                return False, f"总仓位达{self.max_total_pct*100:.0f}%上限"
            if signal.price > 0:
                max_size = int(total_value * self.max_single_pct / signal.price / 100) * 100
                if max_size < 100:
                    return False, "单股仓位不足100股"
            if cash < signal.price * 100:
                return False, "资金不足"
            return True, "买入通过"

        if signal.action == Action.SELL:
            if pos_size <= 0:
                return False, "无持仓"
            # 跌停大概率卖不出
            if self.is_limit_down(pct_change):
                return False, f"跌停卖不出 (跌{pct_change*100:.1f}%)"
            return True, "卖出通过"

        return False, "未知信号"

    def status(self) -> dict:
        return {
            "kill_switch": self.is_stopped(),
            "max_daily_loss": f"{self.max_daily_loss_pct*100:.0f}%",
            "max_single": f"{self.max_single_pct*100:.0f}%",
            "max_total": f"{self.max_total_pct*100:.0f}%",
        }
