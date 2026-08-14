"""
通知模块 — 邮件通知交易信号和异常
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from datetime import datetime

from .logger import log


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_server: str = "smtp.qq.com"
    smtp_port: int = 465
    sender: str = ""
    password: str = ""
    receiver: str = ""


class EmailNotifier:
    """邮件通知（QQ邮箱 / 163 / Gmail 均支持）"""

    def __init__(self, config: EmailConfig):
        self.config = config

    def send(self, subject: str, body: str) -> bool:
        """发送邮件，成功返回 True"""
        if not self.config.enabled:
            log.debug("[通知] 邮件未启用，跳过")
            return False

        if not all([self.config.sender, self.config.password, self.config.receiver]):
            log.warning("[通知] 邮件配置不完整，跳过发送")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.config.sender
            msg["To"] = self.config.receiver
            msg["Subject"] = f"[TradingAgent] {subject}"

            # HTML 格式邮件
            html = f"""
            <html>
            <body style="font-family: monospace; background: #f5f5f5; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white;
                        padding: 20px; border-radius: 8px; border: 1px solid #ddd;">
                <h2 style="color: #333; border-bottom: 2px solid #4CAF50;
                           padding-bottom: 10px;">
                    🤖 TradingAgent
                </h2>
                <p style="color: #666; font-size: 12px;">
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
                <h3 style="color: #333;">{subject}</h3>
                <pre style="background: #f8f8f8; padding: 15px;
                            border-radius: 4px; font-size: 13px;
                            line-height: 1.5; overflow-x: auto;">{body}</pre>
                <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="color: #999; font-size: 11px;">
                    自动交易系统通知 · 请勿直接回复
                </p>
            </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(html, "html", "utf-8"))

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                self.config.smtp_server, self.config.smtp_port, context=context
            ) as server:
                server.login(self.config.sender, self.config.password)
                server.sendmail(self.config.sender, self.config.receiver, msg.as_string())

            log.info(f"[通知] 邮件已发送: {subject}")
            return True

        except Exception as e:
            log.error(f"[通知] 邮件发送失败: {e}")
            return False


class Notifier:
    """统一通知接口"""

    def __init__(self, email_config: EmailConfig):
        self.email = EmailNotifier(email_config)

    def notify_trade(self, action: str, symbol: str, price: float, size: int, reason: str):
        """交易通知"""
        icon = "🟢" if action == "BUY" else "🔴"
        subject = f"{icon} {action} {symbol}"
        body = (
            f"操作: {action}\n"
            f"标的: {symbol}\n"
            f"价格: {price:.2f}\n"
            f"数量: {size}\n"
            f"原因: {reason}\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        log.info(f"[交易] {action} {symbol} @ {price:.2f} x {size} | {reason}")
        self.email.send(subject, body)

    def notify_error(self, error: str):
        """异常通知"""
        log.error(f"[异常] {error}")
        self.email.send("⚠️ 系统异常", error)

    def notify_daily_report(self, report: str):
        """每日报告"""
        log.info("[报告] 每日汇总已生成")
        self.email.send("📊 每日交易报告", report)

    def notify_kill_switch(self):
        """紧急停止通知"""
        log.critical("[紧急] Kill Switch 已触发！所有交易已停止")
        self.email.send("🚨 紧急停止", "Kill Switch 已触发，交易系统已停止运行。")
