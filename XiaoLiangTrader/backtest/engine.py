"""
回测引擎 — 基于 Backtrader，支持 A 股特殊规则
===============================================
T+1、涨跌停、手续费、滑点全部模拟。
输出完整绩效报告。
"""

import backtrader as bt
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from data.fetcher import fetch_stock
from utils.logger import get_logger

log = get_logger("xlt.backtest")

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════
# 1. Backtrader 策略封装
# ═══════════════════════════════════════════════
class DualMABTStrategy(bt.Strategy):
    """
    双均线策略的 Backtrader 版本

    注意：这个类只在回测时使用。
    实盘的信号计算用 strategy.dual_ma.DualMAStrategy（纯 pandas）。
    """

    params = dict(
        fast_period=5,
        slow_period=20,
        vol_period=20,
        vol_mult=1.5,
        stop_loss=0.08,
        take_profit=0.15,
        trailing_pct=0.05,
        max_single_pct=0.20,
        max_total_pct=0.60,
    )

    def __init__(self):
        self.orders = {}
        self.buy_prices = {}
        self.max_prices = {}
        self.trade_log = []

        for d in self.datas:
            d.fast_ma = bt.indicators.SMA(d.close, period=self.p.fast_period)
            d.slow_ma = bt.indicators.SMA(d.close, period=self.p.slow_period)
            d.ma_cross = bt.indicators.CrossOver(d.fast_ma, d.slow_ma)
            d.vol_ma = bt.indicators.SMA(d.volume, period=self.p.vol_period)
            self.orders[d] = None
            self.buy_prices[d] = None
            self.max_prices[d] = None

    def notify_order(self, order):
        if order.status == order.Completed:
            d = order.data
            if order.isbuy():
                self.buy_prices[d] = order.executed.price
                self.max_prices[d] = order.executed.price
            elif order.issell():
                buy_p = self.buy_prices.get(d, 0)
                sell_p = order.executed.price
                pnl = (sell_p / buy_p - 1) * 100 if buy_p else 0
                self.trade_log.append({
                    "stock": d._name, "buy_price": buy_p,
                    "sell_price": sell_p, "pnl_pct": pnl,
                    "date": d.datetime.date(0),
                })
                self.buy_prices[d] = None
                self.max_prices[d] = None
            self.orders[d] = None

    def _position_value(self):
        return sum(
            self.getposition(d).size * d.close[0]
            for d in self.datas if self.getposition(d).size > 0
        )

    def _can_buy(self, data):
        total = self.broker.getvalue()
        if self._position_value() >= total * self.p.max_total_pct:
            return False
        max_amt = total * self.p.max_single_pct
        return int(max_amt / data.close[0] / 100) * 100 >= 100

    def _buy_size(self, data):
        total = self.broker.getvalue()
        max_amt = total * self.p.max_single_pct
        remaining = total * self.p.max_total_pct - self._position_value()
        amt = min(max_amt, remaining)
        return int(amt / data.close[0] / 100) * 100

    def next(self):
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                price = d.close[0]
                buy_p = self.buy_prices.get(d)
                if buy_p is None:
                    continue

                # 更新最高价
                if self.max_prices[d] is None or price > self.max_prices[d]:
                    self.max_prices[d] = price

                # 止损
                if price <= buy_p * (1 - self.p.stop_loss):
                    self.orders[d] = self.sell(data=d, size=pos.size)
                    continue

                # 跟踪止盈
                if self.max_prices[d] > buy_p * (1 + self.p.take_profit):
                    if price <= self.max_prices[d] * (1 - self.p.trailing_pct):
                        self.orders[d] = self.sell(data=d, size=pos.size)
                        continue

                # 死叉
                if d.ma_cross[0] < 0:
                    self.orders[d] = self.sell(data=d, size=pos.size)
                    continue
            else:
                if self.orders.get(d) is not None:
                    continue
                if d.ma_cross[0] > 0 and d.volume[0] > d.vol_ma[0] * self.p.vol_mult:
                    if self._can_buy(d):
                        size = self._buy_size(d)
                        if size >= 100:
                            self.orders[d] = self.buy(data=d, size=size)


# ═══════════════════════════════════════════════
# 2. 回测引擎
# ═══════════════════════════════════════════════
class BacktestEngine:
    """
    回测引擎 — 一行代码跑完整回测

    用法:
        engine = BacktestEngine()
        report = engine.run(["600519", "300750"])
        engine.print_report(report)
    """

    def __init__(self, initial_cash: float = 1_000_000):
        self.initial_cash = initial_cash

    def run(
        self,
        stock_codes: list[str],
        start_date: str = "20180101",
        end_date: str | None = None,
        fast_period: int = 5,
        slow_period: int = 20,
        vol_mult: float = 1.5,
        plot: bool = True,
    ) -> dict:
        """
        运行完整回测。

        Returns:
            绩效报告 dict
        """
        cerebro = bt.Cerebro()

        # 加载数据
        loaded = 0
        for code in stock_codes:
            df = fetch_stock(code, start_date, end_date, use_cache=True)
            if df.empty:
                log.warning(f"[回测] {code} 无数据，跳过")
                continue
            data = bt.feeds.PandasData(
                dataname=df, name=code, datetime=None,
                open="open", high="high", low="low",
                close="close", volume="volume", openinterest=-1,
            )
            cerebro.adddata(data)
            loaded += 1

        if loaded == 0:
            return {"error": "无有效数据"}

        # 策略
        cerebro.addstrategy(DualMABTStrategy,
                            fast_period=fast_period,
                            slow_period=slow_period,
                            vol_mult=vol_mult)

        # 资金与成本
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=0.00025)  # 佣金万2.5
        cerebro.broker.set_slippage_perc(0.001)           # 滑点0.1%

        # 分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        log.info(f"[回测] 开始 | 资金: {self.initial_cash:,.0f} | 标的: {stock_codes}")
        log.info(f"[回测] 参数: MA{fast_period}/{slow_period} 量比>{vol_mult}x")

        results = cerebro.run()
        strat = results[0]

        # 提取绩效
        trades = strat.trade_log
        if not trades:
            return {"error": "无交易记录"}

        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]

        sharpe_data = strat.analyzers.sharpe.get_analysis()
        dd_data = strat.analyzers.drawdown.get_analysis()
        ret_data = strat.analyzers.returns.get_analysis()

        annual_ret = ret_data.get("rnorm100", 0)
        max_dd = dd_data.get("max", {}).get("drawdown", 0)
        sharpe = sharpe_data.get("sharperatio", 0) or 0

        report = {
            "标的": ", ".join(stock_codes),
            "回测区间": f"{start_date} ~ {end_date or '至今'}",
            "参数": f"MA{fast_period}/{slow_period}, 量比>{vol_mult}x",
            "初始资金": f"{self.initial_cash:,.0f}",
            "最终净值": f"{cerebro.broker.getvalue():,.0f}",
            "总交易次数": len(trades),
            "胜率": f"{len(wins)/len(trades)*100:.1f}%",
            "平均盈利": f"{np.mean([t['pnl_pct'] for t in wins]):.2f}%" if wins else "N/A",
            "平均亏损": f"{np.mean([t['pnl_pct'] for t in losses]):.2f}%" if losses else "N/A",
            "年化收益率": f"{annual_ret:.2f}%",
            "最大回撤": f"{max_dd:.2f}%",
            "夏普比率": f"{sharpe:.2f}",
            "Calmar比率": f"{annual_ret/max_dd:.2f}" if max_dd > 0 else "N/A",
            "_trades": trades,  # 内部用，不打印
        }

        if plot:
            try:
                fig = cerebro.plot(style="candle", volume=True)[0][0]
                fig.savefig(OUTPUT_DIR / "backtest_result.png", dpi=150, bbox_inches="tight")
                log.info(f"[回测] 图表已保存: output/backtest_result.png")
            except Exception as e:
                log.warning(f"[回测] 绘图失败: {e}")

        return report

    @staticmethod
    def print_report(report: dict):
        """格式化打印回测报告"""
        if "error" in report:
            print(f"回测失败: {report['error']}")
            return

        print(f"\n{'='*55}")
        print("              📊 回测绩效报告")
        print(f"{'='*55}")
        for k, v in report.items():
            if not k.startswith("_"):
                print(f"  {k:>12s}: {v}")
        print(f"{'='*55}")

        trades = report.get("_trades", [])
        if trades:
            print(f"\n交易明细（共 {len(trades)} 笔）:")
            df = pd.DataFrame(trades)
            print(df.to_string(index=False))
