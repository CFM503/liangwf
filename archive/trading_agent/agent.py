"""
交易 Agent — 编排每日流程
获取数据 → 计算信号 → 风控检查 → 执行订单 → 通知
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 加入项目根目录到 path（用于 import data_fetcher）
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import TradingConfig, load_config
from .strategy import DualMAStrategy, Signal, Action
from .executor import create_executor, BaseExecutor, OrderResult
from .risk import RiskManager
from .notifier import Notifier, EmailConfig
from .logger import log


class TradingAgent:
    """
    自动化交易 Agent

    每日流程:
    1. 初始化（加载配置、连接执行器）
    2. 设置今日风控基准
    3. 遍历标的池:
       a. 获取最新数据
       b. 计算策略信号
       c. 风控校验
       d. 执行订单
       e. 通知
    4. 生成每日报告
    """

    def __init__(self, config: TradingConfig | None = None, config_path: str | None = None):
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        # 初始化各模块
        self.strategy = DualMAStrategy(
            fast_period=self.config.strategy.fast_period,
            slow_period=self.config.strategy.slow_period,
            vol_period=self.config.strategy.vol_period,
            vol_mult=self.config.strategy.vol_mult,
            stop_loss=self.config.strategy.stop_loss,
            take_profit=self.config.strategy.take_profit,
            trailing_pct=self.config.strategy.trailing_pct,
        )

        self.executor = create_executor(
            mode=self.config.broker.mode,
            db_path=self.config.db_path,
            initial_cash=self.config.initial_cash,
        )

        self.risk = RiskManager(
            kill_switch_file=self.config.risk.kill_switch_file,
            max_daily_loss_pct=self.config.risk.max_daily_loss_pct,
            max_single_pct=self.config.risk.max_single_pct,
            max_total_pct=self.config.risk.max_total_pct,
        )

        self.notifier = Notifier(EmailConfig(
            enabled=self.config.email.enabled,
            smtp_server=self.config.email.smtp_server,
            smtp_port=self.config.email.smtp_port,
            sender=self.config.email.sender,
            password=self.config.email.password,
            receiver=self.config.email.receiver,
        ))

    def run_daily(self):
        """每日主流程"""
        log.info("=" * 60)
        log.info("[Agent] 每日交易流程启动")
        log.info(f"[Agent] 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"[Agent] 标的: {self.config.stocks}")
        log.info(f"[Agent] 参数: MA{self.config.strategy.fast_period}"
                 f"/{self.config.strategy.slow_period},"
                 f" 量比>{self.config.strategy.vol_mult}x")
        log.info("=" * 60)

        # 1. Kill Switch 检查
        if self.risk.is_stopped():
            log.critical("[Agent] Kill Switch 已激活，今日流程中止！")
            self.notifier.notify_kill_switch()
            return

        # 2. 设置今日风控基准
        total_value = self.executor.get_total_value()
        self.risk.set_daily_start_value(total_value)
        log.info(f"[Agent] 今日起始净值: {total_value:,.0f}")

        # 3. 遍历标的
        results = []
        for symbol in self.config.stocks:
            try:
                result = self._process_symbol(symbol)
                results.append(result)
            except Exception as e:
                log.error(f"[Agent] 处理 {symbol} 异常: {e}")
                self.notifier.notify_error(f"处理 {symbol} 异常: {e}")

        # 4. 生成报告
        report = self._generate_report(results)
        log.info("\n" + report)
        self.notifier.notify_daily_report(report)

        log.info("[Agent] 每日流程结束")

    def _process_symbol(self, symbol: str) -> dict:
        """处理单只股票"""
        log.info(f"[Agent] ── 处理 {symbol} ──")

        # 获取数据
        from data_fetcher import fetch_stock_data
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")

        df = fetch_stock_data(symbol, start_date, end_date, use_cache=False)
        if df.empty or len(df) < self.config.strategy.slow_period + 2:
            log.warning(f"[Agent] {symbol} 数据不足，跳过")
            return {"symbol": symbol, "signal": "数据不足", "action": "SKIP"}

        # 更新持仓最高价
        current_price = df["close"].iloc[-1]
        pos_size = self.executor.get_position(symbol)
        if pos_size > 0:
            self.executor.update_max_price(symbol, current_price)

        # 计算信号
        signal = self.strategy.get_latest_signal(
            df, symbol,
            position_size=pos_size,
            buy_price=self.executor.get_buy_price(symbol),
            max_price=self.executor.get_max_price(symbol),
        )

        log.info(
            f"[Agent] {symbol} 信号: {signal.action.value} | "
            f"价格: {signal.price:.2f} | "
            f"MA({signal.ma_fast:.2f}/{signal.ma_slow:.2f}) | "
            f"量比: {signal.volume_ratio:.1f}x | "
            f"原因: {signal.reason}"
        )

        if signal.action == Action.HOLD:
            return {"symbol": symbol, "signal": "HOLD", "action": "NONE", "detail": signal.reason}

        # 风控校验
        total_value = self.executor.get_total_value()
        current_cash = self.executor.get_cash()
        pos_value = sum(
            p["size"] * p["buy_price"]
            for p in self.executor.get_positions_summary()
        )

        # 每日亏损检查
        if self.risk.check_daily_loss(total_value):
            self.notifier.notify_error("今日亏损超限，暂停交易")
            return {"symbol": symbol, "signal": signal.action.value, "action": "BLOCKED",
                    "detail": "每日亏损超限"}

        valid, reason = self.risk.validate_signal(
            signal, current_cash, pos_value, total_value, pos_size
        )
        if not valid:
            log.warning(f"[Agent] {symbol} 信号被风控拒绝: {reason}")
            return {"symbol": symbol, "signal": signal.action.value, "action": "BLOCKED",
                    "detail": reason}

        # 执行订单
        result = self.executor.execute(signal)
        if result.success:
            self.notifier.notify_trade(
                result.action, result.symbol, result.price, result.size, result.reason
            )
            return {"symbol": symbol, "signal": signal.action.value, "action": "EXECUTED",
                    "detail": f"{result.action} @ {result.price:.2f} x {result.size}"}
        else:
            log.warning(f"[Agent] {symbol} 执行失败: {result.message}")
            return {"symbol": symbol, "signal": signal.action.value, "action": "FAILED",
                    "detail": result.message}

    def _generate_report(self, results: list[dict]) -> str:
        """生成每日报告"""
        total_value = self.executor.get_total_value()
        cash = self.executor.get_cash()
        positions = self.executor.get_positions_summary()

        lines = [
            f"{'='*50}",
            f"📊 每日交易报告 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"{'='*50}",
            f"",
            f"💰 账户概况:",
            f"  总净值:   {total_value:>12,.0f}",
            f"  可用资金: {cash:>12,.0f}",
            f"  持仓市值: {total_value - cash:>12,.0f}",
            f"  仓位占比: {(total_value - cash) / total_value * 100:.1f}%",
            f"",
        ]

        if positions:
            lines.append("📦 当前持仓:")
            for p in positions:
                lines.append(
                    f"  {p['symbol']}: {p['size']}股 "
                    f"@ {p['buy_price']:.2f} "
                    f"(最高 {p['max_price']:.2f})"
                )
            lines.append("")

        if results:
            lines.append("📋 今日操作:")
            for r in results:
                icon = {"EXECUTED": "✅", "BLOCKED": "🚫", "FAILED": "❌",
                        "NONE": "⬜", "SKIP": "⏭️"}.get(r["action"], "❓")
                lines.append(
                    f"  {icon} {r['symbol']}: {r['signal']} → {r['action']} | {r['detail']}"
                )

        # 风控状态
        risk_status = self.risk.status()
        lines.extend([
            "",
            "🛡️ 风控状态:",
            f"  Kill Switch: {'🔴 激活' if risk_status['kill_switch_active'] else '🟢 正常'}",
            f"  每日亏损上限: {risk_status['max_daily_loss']}",
            f"  单股仓位上限: {risk_status['max_single_position']}",
            f"  总仓位上限: {risk_status['max_total_position']}",
        ])

        return "\n".join(lines)

    def get_status(self) -> dict:
        """获取 Agent 完整状态"""
        return {
            "time": datetime.now().isoformat(),
            "config": {
                "stocks": self.config.stocks,
                "strategy": f"MA{self.config.strategy.fast_period}/{self.config.strategy.slow_period}",
                "broker_mode": self.config.broker.mode,
            },
            "account": {
                "cash": self.executor.get_cash(),
                "total_value": self.executor.get_total_value(),
                "positions": self.executor.get_positions_summary(),
            },
            "risk": self.risk.status(),
            "recent_orders": self.executor.get_recent_orders(10),
        }
