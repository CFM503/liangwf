"""
双均线交叉策略 - Backtrader 实现
===================================
策略逻辑：
  - MA5 上穿 MA20（金叉）+ 成交量放大 → 买入
  - MA5 下穿 MA20（死叉）或 触发止损/止盈 → 卖出
  - 仓位控制：单股 ≤ 20%，总仓位 ≤ 60%
  - 止损 -8%，固定止盈 +15%，跟踪止盈（最高回撤 5%）
"""

import datetime
import backtrader as bt
import matplotlib

matplotlib.use("Agg")  # 无头模式，不弹窗
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_fetcher import fetch_stock_data, DEFAULT_STOCKS


# ═══════════════════════════════════════════════
# 1. 策略类
# ═══════════════════════════════════════════════
class DualMAStrategy(bt.Strategy):
    params = dict(
        fast_period=5,       # 短期均线
        slow_period=20,      # 长期均线
        vol_period=20,       # 均量周期
        vol_mult=1.5,        # 成交量倍数阈值
        stop_loss=0.08,      # 止损比例 8%
        take_profit=0.15,    # 止盈比例 15%
        trailing_pct=0.05,   # 跟踪止盈回撤 5%
        max_single_pct=0.20, # 单股最大仓位 20%
        max_total_pct=0.60,  # 总仓位上限 60%
    )

    def __init__(self):
        self.orders = {}          # 每只股票的待执行订单
        self.buy_prices = {}      # 每只股票的买入价
        self.max_prices = {}      # 每只股票持仓期最高价
        self.trade_log = []       # 交易记录

        for d in self.datas:
            # 均线
            d.fast_ma = bt.indicators.SMA(d.close, period=self.p.fast_period)
            d.slow_ma = bt.indicators.SMA(d.close, period=self.p.slow_period)
            d.ma_cross = bt.indicators.CrossOver(d.fast_ma, d.slow_ma)

            # 均量
            d.vol_ma = bt.indicators.SMA(d.volume, period=self.p.vol_period)

            self.orders[d] = None
            self.buy_prices[d] = None
            self.max_prices[d] = None

    def log(self, msg, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # 静默模式下不打印（可通过注释开启）
        # print(f"[{dt}] {msg}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            d = order.data
            if order.isbuy():
                self.buy_prices[d] = order.executed.price
                self.max_prices[d] = order.executed.price
                self.log(
                    f"买入 {d._name} @ {order.executed.price:.2f} "
                    f"x {order.executed.size:.0f}"
                )
            elif order.issell():
                buy_p = self.buy_prices.get(d, 0)
                sell_p = order.executed.price
                pnl_pct = (sell_p / buy_p - 1) * 100 if buy_p else 0
                self.trade_log.append(
                    {
                        "stock": d._name,
                        "buy_price": buy_p,
                        "sell_price": sell_p,
                        "pnl_pct": pnl_pct,
                        "date": d.datetime.date(0),
                    }
                )
                self.buy_prices[d] = None
                self.max_prices[d] = None
                self.log(
                    f"卖出 {d._name} @ {sell_p:.2f} "
                    f"盈亏 {pnl_pct:+.2f}%"
                )
            self.orders[d] = None

    def _get_total_position_value(self):
        """当前所有持仓市值"""
        return sum(
            self.getposition(d).size * d.close[0]
            for d in self.datas
            if self.getposition(d).size > 0
        )

    def _can_buy(self, data):
        """仓位控制检查"""
        total_value = self.broker.getvalue()
        # 总仓位检查
        current_pos = self._get_total_position_value()
        if current_pos >= total_value * self.p.max_total_pct:
            return False
        # 单股仓位检查
        max_amount = total_value * self.p.max_single_pct
        price = data.close[0]
        if price <= 0:
            return False
        max_shares = int(max_amount / price / 100) * 100  # A股100股整数
        return max_shares >= 100

    def _calc_buy_size(self, data):
        """计算买入数量（100股整数）"""
        total_value = self.broker.getvalue()
        max_amount = total_value * self.p.max_single_pct
        # 同时不超过总仓位限制
        current_pos = self._get_total_position_value()
        remaining = total_value * self.p.max_total_pct - current_pos
        amount = min(max_amount, remaining)
        price = data.close[0]
        if price <= 0:
            return 0
        shares = int(amount / price / 100) * 100
        return max(shares, 0)

    def next(self):
        for d in self.datas:
            pos = self.getposition(d)

            # ── 持仓中：检查止损/止盈/跟踪止盈 ──
            if pos.size > 0:
                price = d.close[0]
                buy_p = self.buy_prices.get(d)
                if buy_p is None:
                    continue

                # 更新持仓最高价
                if self.max_prices[d] is None or price > self.max_prices[d]:
                    self.max_prices[d] = price

                # 止损
                if price <= buy_p * (1 - self.p.stop_loss):
                    self.orders[d] = self.sell(data=d, size=pos.size)
                    self.log(f"止损 {d._name}")
                    continue

                # 固定止盈
                if price >= buy_p * (1 + self.p.take_profit):
                    # 切换到跟踪止盈模式（max_prices 已在跟踪）
                    pass

                # 跟踪止盈：从最高点回撤超过阈值
                if self.max_prices[d] > buy_p * (1 + self.p.take_profit):
                    if price <= self.max_prices[d] * (1 - self.p.trailing_pct):
                        self.orders[d] = self.sell(data=d, size=pos.size)
                        self.log(f"跟踪止盈 {d._name}")
                        continue

                # 死叉卖出
                if d.ma_cross[0] < 0:
                    self.orders[d] = self.sell(data=d, size=pos.size)
                    self.log(f"死叉卖出 {d._name}")
                    continue

            # ── 空仓：检查买入信号 ──
            else:
                if self.orders.get(d) is not None:
                    continue

                # 金叉 + 成交量放大
                is_golden_cross = d.ma_cross[0] > 0
                vol_ok = d.volume[0] > d.vol_ma[0] * self.p.vol_mult

                if is_golden_cross and vol_ok and self._can_buy(d):
                    size = self._calc_buy_size(d)
                    if size >= 100:
                        self.orders[d] = self.buy(data=d, size=size)


# ═══════════════════════════════════════════════
# 2. 绩效分析
# ═══════════════════════════════════════════════
class PerformanceAnalyzer:
    """从 Backtrader 的分析器提取核心指标"""

    @staticmethod
    def analyze(cerebro, strategy):
        strat = strategy[0]

        # 基础指标
        total_return = cerebro.broker.getvalue() / 1000000 - 1  # 假设初始100万

        # 交易统计
        trades = strat.trade_log
        if not trades:
            print("无交易记录")
            return {}

        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]

        # 使用 backtrader 内置分析器
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()

        annual_return = returns.get("rnorm100", 0)
        max_dd = drawdown.get("max", {}).get("drawdown", 0)
        sharpe_ratio = sharpe.get("sharperatio", 0) or 0

        # 年化计算（手动校验）
        days = (trades[-1]["date"] - trades[0]["date"]).days if len(trades) > 1 else 365
        years = max(days / 365.25, 0.01)

        results = {
            "总交易次数": len(trades),
            "盈利次数": len(wins),
            "亏损次数": len(losses),
            "胜率": f"{len(wins) / len(trades) * 100:.1f}%",
            "平均盈利": f"{np.mean([t['pnl_pct'] for t in wins]):.2f}%" if wins else "N/A",
            "平均亏损": f"{np.mean([t['pnl_pct'] for t in losses]):.2f}%" if losses else "N/A",
            "盈亏比": (
                f"{abs(np.mean([t['pnl_pct'] for t in wins]) / np.mean([t['pnl_pct'] for t in losses])):.2f}"
                if wins and losses else "N/A"
            ),
            "年化收益率": f"{annual_return:.2f}%",
            "最大回撤": f"{max_dd:.2f}%",
            "夏普比率": f"{sharpe_ratio:.2f}",
            "Calmar比率": (
                f"{annual_return / max_dd:.2f}" if max_dd > 0 else "N/A"
            ),
            "最终净值": f"{cerebro.broker.getvalue():,.0f}",
        }
        return results


# ═══════════════════════════════════════════════
# 3. 回测主流程
# ═══════════════════════════════════════════════
def run_backtest(
    stock_codes: list[str] | None = None,
    start_date: str = "20180101",
    end_date: str = "20251231",
    initial_cash: float = 1_000_000,
    fast_period: int = 5,
    slow_period: int = 20,
    vol_mult: float = 1.5,
    plot: bool = True,
) -> dict:
    """
    运行完整回测

    Args:
        stock_codes: 股票代码列表，默认使用 DEFAULT_STOCKS
        start_date: 开始日期
        end_date: 结束日期
        initial_cash: 初始资金
        fast_period: 短期均线周期
        slow_period: 长期均线周期
        vol_mult: 成交量倍数
        plot: 是否输出图表

    Returns:
        绩效指标字典
    """
    if stock_codes is None:
        stock_codes = list(DEFAULT_STOCKS.keys())[:3]  # 默认3只

    cerebro = bt.Cerebro()

    # 加载数据
    for code in stock_codes:
        df = fetch_stock_data(code, start_date, end_date)
        if df.empty:
            print(f"[跳过] {code} 无数据")
            continue
        data = bt.feeds.PandasData(
            dataname=df,
            name=code,
            datetime=None,  # 使用 index
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
        cerebro.adddata(data)

    # 策略参数
    cerebro.addstrategy(
        DualMAStrategy,
        fast_period=fast_period,
        slow_period=slow_period,
        vol_mult=vol_mult,
    )

    # 初始资金
    cerebro.broker.setcash(initial_cash)
    # 手续费：万2.5 双边 + 印花税 千1（卖出）
    cerebro.broker.setcommission(commission=0.00025)
    # A股最小单位
    cerebro.broker.set_slippage_perc(0.001)  # 0.1% 滑点

    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    print(f"{'='*50}")
    print(f"回测开始 | 资金: {initial_cash:,.0f}")
    print(f"标的: {', '.join(stock_codes)}")
    print(f"参数: MA{fast_period}/{slow_period}, 量比>{vol_mult}x")
    print(f"{'='*50}")

    results = cerebro.run()
    strat = results[0]

    # 输出绩效
    perf = PerformanceAnalyzer.analyze(cerebro, results)
    print(f"\n{'='*50}")
    print("回测绩效报告")
    print(f"{'='*50}")
    for k, v in perf.items():
        print(f"  {k:>12s}: {v}")
    print(f"{'='*50}")

    # 输出交易明细
    if strat.trade_log:
        trade_df = pd.DataFrame(strat.trade_log)
        print(f"\n交易明细（共 {len(trade_df)} 笔）:")
        print(trade_df.to_string(index=False))

    # 绘图
    if plot:
        fig = cerebro.plot(style="candle", volume=True)[0][0]
        fig.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
        print("\n图表已保存: backtest_result.png")

    return perf


# ═══════════════════════════════════════════════
# 4. 参数扫描
# ═══════════════════════════════════════════════
def parameter_scan(
    stock_codes: list[str] | None = None,
):
    """
    网格搜索最优参数组合
    fast: 3-10, slow: 15-30, vol_mult: 1.0-2.0
    """
    if stock_codes is None:
        stock_codes = list(DEFAULT_STOCKS.keys())[:2]  # 扫描时用2只加速

    best = {"sharpe": -999, "params": None, "perf": None}
    results_list = []

    for fast in range(3, 11):
        for slow in range(15, 31, 5):
            if fast >= slow:
                continue
            for vol_mult in [1.0, 1.3, 1.5, 1.8, 2.0]:
                try:
                    perf = run_backtest(
                        stock_codes=stock_codes,
                        fast_period=fast,
                        slow_period=slow,
                        vol_mult=vol_mult,
                        plot=False,
                    )
                    sharpe_str = perf.get("夏普比率", "0")
                    sharpe = float(sharpe_str) if sharpe_str != "N/A" else 0

                    row = {
                        "fast": fast,
                        "slow": slow,
                        "vol_mult": vol_mult,
                        "sharpe": sharpe,
                        "annual_return": perf.get("年化收益率", "N/A"),
                        "max_drawdown": perf.get("最大回撤", "N/A"),
                        "win_rate": perf.get("胜率", "N/A"),
                    }
                    results_list.append(row)

                    if sharpe > best["sharpe"]:
                        best["sharpe"] = sharpe
                        best["params"] = (fast, slow, vol_mult)
                        best["perf"] = perf
                except Exception as e:
                    print(f"[扫描错误] MA{fast}/{slow} vol={vol_mult}: {e}")

    # 输出结果
    if results_list:
        df = pd.DataFrame(results_list)
        df = df.sort_values("sharpe", ascending=False)
        print(f"\n{'='*60}")
        print("参数扫描结果 TOP 10:")
        print(f"{'='*60}")
        print(df.head(10).to_string(index=False))

    if best["params"]:
        print(f"\n最优参数: MA{best['params'][0]}/{best['params'][1]}, "
              f"量比>{best['params'][2]}x, 夏普={best['sharpe']:.2f}")
        print("绩效:")
        for k, v in best["perf"].items():
            print(f"  {k}: {v}")

    return results_list


# ═══════════════════════════════════════════════
# 5. 入口
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "scan":
        parameter_scan()
    else:
        run_backtest()
