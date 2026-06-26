"""
交易 Agent — 每日自动编排
============================
获取数据 → 计算信号 → 风控 → 执行 → 通知 → 报告
这就是每天下午 3:10 自动运行的主流程。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config.settings import Config
from data.fetcher import fetch_stock, STOCK_NAMES
from strategy.signals import Action
from strategy.dual_ma import DualMAStrategy
from strategy.ml_strategy import MLEnhancedStrategy
from ml_model.predictor import MLPredictor
from bot.executor import SimulatorExecutor, create_executor
from bot.risk import RiskManager
from bot.notifier import Notifier
from utils.logger import get_logger

log = get_logger("xlt.agent")


class TradingAgent:
    """
    自动化交易 Agent

    用法:
        agent = TradingAgent(config)
        agent.run_daily()
    """

    def __init__(self, config: Config):
        self.config = config

        # 基础策略
        self.base_strategy = DualMAStrategy(
            fast_period=config.strategy.fast_period,
            slow_period=config.strategy.slow_period,
            vol_period=config.strategy.vol_period,
            vol_mult=config.strategy.vol_mult,
            stop_loss=config.strategy.stop_loss,
            take_profit=config.strategy.take_profit,
            trailing_pct=config.strategy.trailing_pct,
        )

        # ML 增强（可选）
        self.predictor = None
        if config.ml.enabled:
            self.predictor = MLPredictor(
                model_type=config.ml.model_type,
                forward_days=config.ml.forward_days,
                threshold=config.ml.threshold,
                n_estimators=config.ml.n_estimators,
                max_depth=config.ml.max_depth,
            )
            models = self.predictor.list_saved_models()
            if models:
                self.predictor.load(models[-1])
                log.info(f"[Agent] 已加载 ML 模型: {models[-1]}")
            else:
                log.warning("[Agent] ML 已启用但无模型，请先运行 --train")
                self.predictor = None

        # 组合策略
        if self.predictor:
            self.strategy = MLEnhancedStrategy(
                base_strategy=self.base_strategy,
                ml_confidence=config.ml.confidence_threshold,
                ml_predictor=self.predictor,
            )
        else:
            self.strategy = self.base_strategy

        # 执行器
        self.executor = create_executor(
            mode=config.broker.mode,
            db_path=config.db_path,
            initial_cash=config.initial_cash,
        )

        # 风控
        self.risk = RiskManager(
            kill_switch_file=config.risk.kill_switch_file,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_single_pct=config.risk.max_single_pct,
            max_total_pct=config.risk.max_total_pct,
            limit_up=config.risk.limit_up_threshold,
            limit_down=config.risk.limit_down_threshold,
        )

        # 通知
        self.notifier = Notifier(
            enabled=config.email.enabled,
            smtp_server=config.email.smtp_server,
            smtp_port=config.email.smtp_port,
            sender=config.email.sender,
            password=config.email.password,
            receiver=config.email.receiver,
        )

    def run_daily(self):
        """每日主流程"""
        log.info("=" * 60)
        log.info("[Agent] 每日流程启动")
        log.info(f"[Agent] {datetime.now():%Y-%m-%d %H:%M}")
        log.info(f"[Agent] 标的: {self.config.stocks}")
        log.info(f"[Agent] 参数: MA{self.config.strategy.fast_period}"
                 f"/{self.config.strategy.slow_period}")
        log.info(f"[Agent] ML: {'✓ ' + self.config.ml.model_type if self.config.ml.enabled else '✗'}")
        log.info("=" * 60)

        # T+1 重置
        if hasattr(self.executor, 'reset_today'):
            self.executor.reset_today()

        # Kill Switch
        if self.risk.is_stopped():
            log.critical("[Agent] Kill Switch 激活，中止！")
            self.notifier.notify_kill_switch()
            return

        # 风控基准
        total = self.executor.get_total_value()
        self.risk.set_daily_start_value(total)
        log.info(f"[Agent] 今日起始净值: {total:,.0f}")

        # 遍历标的
        results = []
        for symbol in self.config.stocks:
            try:
                results.append(self._process_symbol(symbol))
            except Exception as e:
                log.error(f"[Agent] {symbol} 异常: {e}")
                self.notifier.notify_error(f"{symbol}: {e}")

        # 每日报告
        report = self._make_report(results)
        log.info("\n" + report)
        self.notifier.notify_report(report)
        log.info("[Agent] 流程结束")

    def _process_symbol(self, symbol: str) -> dict:
        """处理单只股票"""
        log.info(f"[Agent] ── {symbol} ({STOCK_NAMES.get(symbol, '')}) ──")

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        df = fetch_stock(symbol, start, end, use_cache=False)

        if df.empty or len(df) < self.config.strategy.slow_period + 2:
            log.warning(f"[Agent] {symbol} 数据不足")
            return {"symbol": symbol, "signal": "数据不足", "action": "SKIP"}

        pct_change = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) if len(df) >= 2 else 0

        price = df["close"].iloc[-1]
        pos_size = self.executor.get_position(symbol)
        if pos_size > 0:
            self.executor.update_max_price(symbol, price)

        # 计算信号
        sig = self.strategy.get_latest_signal(
            df, symbol, pos_size,
            self.executor.get_buy_price(symbol),
            self.executor.get_max_price(symbol),
        )

        log.info(f"[Agent] {symbol} {sig.action.value} | {price:.2f} | "
                 f"MA({sig.ma_fast:.1f}/{sig.ma_slow:.1f}) | {sig.reason}")

        if sig.action == Action.HOLD:
            return {"symbol": symbol, "signal": "HOLD", "action": "NONE", "detail": sig.reason}

        # 风控
        total = self.executor.get_total_value()
        cash = self.executor.get_cash()
        pos_val = sum(p["size"] * p["buy_price"] for p in self.executor.get_positions_summary())

        if self.risk.check_daily_loss(total):
            self.notifier.notify_error("今日亏损超限")
            return {"symbol": symbol, "signal": sig.action.value, "action": "BLOCKED", "detail": "亏损超限"}

        ok, reason = self.risk.validate_signal(sig, cash, pos_val, total, pos_size, pct_change)
        if not ok:
            log.warning(f"[Agent] {symbol} 风控拒绝: {reason}")
            return {"symbol": symbol, "signal": sig.action.value, "action": "BLOCKED", "detail": reason}

        # 执行
        result = self.executor.execute(sig)
        if result.success:
            self.notifier.notify_trade(result.action, result.symbol, result.price, result.size, result.reason)
            return {"symbol": symbol, "signal": sig.action.value, "action": "EXECUTED",
                    "detail": f"{result.action} @ {result.price:.2f} x {result.size}"}
        else:
            return {"symbol": symbol, "signal": sig.action.value, "action": "FAILED", "detail": result.message}

    def _make_report(self, results: list[dict]) -> str:
        total = self.executor.get_total_value()
        cash = self.executor.get_cash()
        positions = self.executor.get_positions_summary()
        rs = self.risk.status()

        lines = [
            f"{'='*50}",
            f"📊 每日报告 {datetime.now():%Y-%m-%d %H:%M}",
            f"{'='*50}", "",
            f"💰 账户:",
            f"  总净值:   {total:>12,.0f}",
            f"  可用资金: {cash:>12,.0f}",
            f"  持仓市值: {total-cash:>12,.0f}",
            f"  仓位:     {(total-cash)/total*100:.1f}%", "",
        ]

        if positions:
            lines.append("📦 持仓:")
            for p in positions:
                lines.append(f"  {p['symbol']}: {p['size']}股 @ {p['buy_price']:.2f}")
            lines.append("")

        lines.append("📋 操作:")
        for r in results:
            icon = {"EXECUTED": "✅", "BLOCKED": "🚫", "FAILED": "❌", "NONE": "⬜", "SKIP": "⏭️"}.get(r["action"], "❓")
            lines.append(f"  {icon} {r['symbol']}: {r['detail']}")

        lines.extend(["", "🛡️ 风控:", f"  Kill Switch: {'🔴' if rs['kill_switch'] else '🟢'}"])

        return "\n".join(lines)

    def get_status(self) -> dict:
        return {
            "time": datetime.now().isoformat(),
            "stocks": self.config.stocks,
            "strategy": f"MA{self.config.strategy.fast_period}/{self.config.strategy.slow_period}",
            "ml_enabled": self.config.ml.enabled,
            "ml_model_type": self.config.ml.model_type if self.config.ml.enabled else "none",
            "cash": self.executor.get_cash(),
            "total_value": self.executor.get_total_value(),
            "positions": self.executor.get_positions_summary(),
            "risk": self.risk.status(),
        }
