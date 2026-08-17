"""
钉钉通知模块（预留接口占位）
============================
为后续接入钉钉机器人或钉钉多维表格预留的可插拔实现。
当前版本保持未实现状态，不包含任何猜测性 API 调用代码。
"""

from typing import Optional
from .notifier import BaseNotifier
from utils.logger import get_logger

log = get_logger("xlt.dingtalk")


class DingTalkNotifier(BaseNotifier):
    """
    钉钉通知实现（预留占位）
    """

    def __init__(
        self,
        enabled: bool = False,
        app_key: str = "",
        app_secret: str = "",
        table_id: str = "",
        webhook_url: str = "",
        secret: str = "",
    ):
        self.enabled = enabled
        self.app_key = app_key
        self.app_secret = app_secret
        self.table_id = table_id
        self.webhook_url = webhook_url
        self.secret = secret

    def notify_report(self, report: str) -> bool:
        """发送每日报告至钉钉"""
        if not self.enabled:
            return False
        log.info("[钉钉通知] 钉钉多维表格/机器人通知尚未实现，跳过发送")
        return False

    def notify_trade(self, action: str, symbol: str, price: float, size: int, reason: str) -> bool:
        """发送交易信号至钉钉"""
        if not self.enabled:
            return False
        log.info("[钉钉通知] 钉钉交易信号推送尚未实现，跳过发送")
        return False

    def notify_error(self, error: str) -> bool:
        """发送异常告警至钉钉"""
        if not self.enabled:
            return False
        log.info("[钉钉通知] 钉钉异常告警尚未实现，跳过发送")
        return False

    def notify_kill_switch(self) -> bool:
        """发送紧急停止至钉钉"""
        if not self.enabled:
            return False
        log.info("[钉钉通知] 钉钉紧急停止通知尚未实现，跳过发送")
        return False
