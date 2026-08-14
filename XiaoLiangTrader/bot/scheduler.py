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
from strategy.signals import Action, Signal
from scripts.daily_signal import DailySignalPipeline
from bot.executor import SimulatorExecutor, create_executor
from bot.risk import RiskManager
from bot.notifier import Notifier
from utils.logger import get_logger

log = get_logger("xlt.agent")


class TradingAgent:
    """
    自动化交易 Agent — 基于统一的 DailySignalPipeline 进行信号生成
    """

    def __init__(self, config: Config):
        self.config = config

        # 统一信号流水线
        self.pipeline = DailySignalPipeline(
            ml_confidence=config.ml.confidence_threshold,
            fast_period=config.strategy.fast_period,
            slow_period=config.strategy.slow_period,
            vol_mult=config.strategy.vol_mult,
            stop_loss_pct=config.strategy.stop_loss,
            take_profit_pct=config.strategy.take_profit,
        )

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
        """每日主流程 — 使用统一的 DailySignalPipeline 进行信号生成"""
        log.info("=" * 60)
        log.info("[Agent] 每日流程启动")
        log.info(f"[Agent] {datetime.now():%Y-%m-%d %H:%M}")
        log.info(f"[Agent] 标的规模: {len(self.config.stocks)} 只")
        log.info(f"[Agent] 策略引擎: MA{self.config.strategy.fast_period}/{self.config.strategy.slow_period}")
        log.info(f"[Agent] ML引擎: LightGBM (置信度门槛 ≥ {self.config.ml.confidence_threshold:.2f})")
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

        # 1. 统一调用 DailySignalPipeline 生成全市场/股票池信号
        payload = self.pipeline.scan_signals(self.config.stocks)
        signals_map = {s["symbol"]: s for s in payload.get("signals", [])}
        log.info(f"[Agent] 统一流水线扫描完成: 触发信号 {len(signals_map)} 条 (BUY: {payload.get('buy_count', 0)}, SELL: {payload.get('sell_count', 0)})")

        # 2. 结合持仓与风控逐一执行
        results = []
        for symbol in self.config.stocks:
            try:
                sig_info = signals_map.get(symbol)
                results.append(self._process_symbol_with_signal(symbol, sig_info))
            except Exception as e:
                log.error(f"[Agent] {symbol} 异常: {e}")
                self.notifier.notify_error(f"{symbol}: {e}")

        # 3. 生成并发送每日报告
        report = self._make_report(results, payload)
        log.info("\n" + report)
        self.notifier.notify_report(report)
        log.info("[Agent] 流程结束")

    def _process_symbol_with_signal(self, symbol: str, sig_info: dict | None) -> dict:
        """结合流水线信号与持仓风控状态处理单只股票"""
        pos_size = self.executor.get_position(symbol)
        
        # 无流水线信号时
        if sig_info is None:
            return {"symbol": symbol, "signal": "HOLD", "action": "NONE", "detail": "无触发信号 (观望)"}

        action_str = sig_info["action"]
        price = sig_info["price"]
        reason = sig_info["reason"]

        if pos_size > 0:
            self.executor.update_max_price(symbol, price)

        # 构造统一 Signal 对象
        if action_str == "BUY":
            # 如果已有持仓，不重复买入
            if pos_size > 0:
                return {"symbol": symbol, "signal": "BUY_IGNORED", "action": "NONE", "detail": "已持有该标的，跳过加仓"}
            sig = Signal(Action.BUY, symbol, price, 0, reason=reason, ml_score=sig_info.get("ml_confidence", -1.0))
        elif action_str == "SELL":
            # 如果空仓，不产生无持仓卖出操作
            if pos_size == 0:
                return {"symbol": symbol, "signal": "SELL_SIGNAL", "action": "NONE", "detail": f"出现卖出信号但当前未持仓 ({reason})"}
            sig = Signal(Action.SELL, symbol, price, pos_size, reason=reason, ml_score=sig_info.get("ml_confidence", -1.0))
        else:
            return {"symbol": symbol, "signal": "HOLD", "action": "NONE", "detail": reason}

        # ── 风控校验 ──
        total = self.executor.get_total_value()
        cash = self.executor.get_cash()
        pos_val = sum(p["size"] * p["buy_price"] for p in self.executor.get_positions_summary())

        if self.risk.check_daily_loss(total):
            self.notifier.notify_error("今日亏损超限")
            return {"symbol": symbol, "signal": sig.action.value, "action": "BLOCKED", "detail": "亏损超限"}

        ok, risk_reason = self.risk.validate_signal(sig, cash, pos_val, total, pos_size, 0.0)
        if not ok:
            log.warning(f"[Agent] {symbol} 风控拒绝: {risk_reason}")
            return {"symbol": symbol, "signal": sig.action.value, "action": "BLOCKED", "detail": risk_reason}

        # ── 执行操作 ──
        result = self.executor.execute(sig)
        if result.success:
            self.notifier.notify_trade(result.action, result.symbol, result.price, result.size, result.reason)
            return {"symbol": symbol, "signal": sig.action.value, "action": "EXECUTED",
                    "detail": f"{result.action} @ {result.price:.2f} x {result.size} ({result.reason})"}
        else:
            return {"symbol": symbol, "signal": sig.action.value, "action": "FAILED", "detail": result.message}

    def _make_report(self, results: list[dict], payload: dict | None = None) -> str:
        total = self.executor.get_total_value()
        cash = self.executor.get_cash()
        positions = self.executor.get_positions_summary()
        rs = self.risk.status()

        lines = [
            f"{'='*50}",
            f"📊 XiaoLiangTrader 每日运行报告 {datetime.now():%Y-%m-%d %H:%M}",
            f"{'='*50}", "",
            f"💰 账户资产:",
            f"  总净值:   {total:>12,.0f}",
            f"  可用资金: {cash:>12,.0f}",
            f"  持仓市值: {total-cash:>12,.0f}",
            f"  仓位:     {(total-cash)/total*100:.1f}%", "",
        ]

        if positions:
            lines.append("📦 当前持仓:")
            for p in positions:
                lines.append(f"  {p['symbol']} ({STOCK_NAMES.get(p['symbol'], '')}): {p['size']}股 @ {p['buy_price']:.2f}")
            lines.append("")

        lines.append("📋 触发操作与信号明细:")
        executed_or_signal = [r for r in results if r["action"] != "NONE"]
        if not executed_or_signal:
            lines.append("  今日无交易操作执行。")
        else:
            for r in executed_or_signal:
                icon = {"EXECUTED": "✅", "BLOCKED": "🚫", "FAILED": "❌"}.get(r["action"], "ℹ️")
                lines.append(f"  {icon} {r['symbol']}: {r['detail']}")

        lines.extend([
            "",
            "🛡️ 风控状态:",
            f"  Kill Switch: {'🔴 触发停止' if rs['kill_switch'] else '🟢 正常运行'}",
            "",
            f"⚠️ 声明：{payload.get('disclaimer') if payload else '仅供量化研究参考，不构成投资建议。'}",
        ])

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
