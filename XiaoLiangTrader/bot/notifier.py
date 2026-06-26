"""
邮件通知 — 交易信号、异常、每日报告
=====================================
QQ邮箱 / 163 / Gmail 都支持。
需要在 config.yaml 中配置 SMTP 信息。
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from utils.logger import get_logger

log = get_logger("xlt.notify")


class Notifier:
    def __init__(
        self,
        enabled: bool = False,
        smtp_server: str = "smtp.qq.com",
        smtp_port: int = 465,
        sender: str = "",
        password: str = "",
        receiver: str = "",
    ):
        self.enabled = enabled
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.receiver = receiver

    def _send(self, subject: str, body: str) -> bool:
        if not self.enabled:
            return False
        if not all([self.sender, self.password, self.receiver]):
            log.warning("[通知] 邮件配置不完整")
            return False
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender
            msg["To"] = self.receiver
            msg["Subject"] = f"[XiaoLiang] {subject}"

            html = f"""<html><body style="font-family:monospace;padding:20px">
            <h2 style="color:#333">🏫 XiaoLiangTrader</h2>
            <p style="color:#999">{datetime.now():%Y-%m-%d %H:%M}</p>
            <h3>{subject}</h3>
            <pre style="background:#f8f8f8;padding:15px;border-radius:4px">{body}</pre>
            <hr><p style="color:#ccc;font-size:11px">自动交易通知</p>
            </body></html>"""
            msg.attach(MIMEText(html, "html", "utf-8"))

            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=ctx) as s:
                s.login(self.sender, self.password)
                s.sendmail(self.sender, self.receiver, msg.as_string())
            log.info(f"[通知] 邮件已发送: {subject}")
            return True
        except Exception as e:
            log.error(f"[通知] 发送失败: {e}")
            return False

    def notify_trade(self, action: str, symbol: str, price: float, size: int, reason: str):
        icon = "🟢" if action == "BUY" else "🔴"
        body = f"操作: {action}\n标的: {symbol}\n价格: {price:.2f}\n数量: {size}\n原因: {reason}"
        log.info(f"[交易] {action} {symbol} @ {price:.2f} x {size} | {reason}")
        self._send(f"{icon} {action} {symbol}", body)

    def notify_error(self, error: str):
        log.error(f"[异常] {error}")
        self._send("⚠️ 异常", error)

    def notify_report(self, report: str):
        log.info("[报告] 每日汇总")
        self._send("📊 每日报告", report)

    def notify_kill_switch(self):
        log.critical("[紧急] Kill Switch!")
        self._send("🚨 紧急停止", "Kill Switch 已触发，交易系统已停止。")
