"""
回测引擎 + 绩效分析
计算: 年化收益、夏普比率、最大回撤、胜率、盈亏比、Calmar比率等
输出: 绩效报告 + 净值曲线图
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无GUI后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

from strategy import create_cerebro, load_data_to_cerebro, MACrossStrategy
from data_generator import generate_all_stocks


# ── 中文字体 ──
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


class PerformanceAnalyzer:
    """计算并展示回测绩效"""

    def __init__(self, cerebro: "bt.Cerebro", strategy: MACrossStrategy):
        self.cerebro = cerebro
        self.strategy = strategy
        self.initial_cash = cerebro.broker.startingcash
        self.final_value = cerebro.broker.getvalue()

    def get_equity_curve(self) -> pd.DataFrame:
        """提取每日净值序列"""
        # 通过 observers 提取
        try:
            # backtrader 的 observers 数据
            strat = self.strategy
            # 用 analyzers 获取
            return self._equity_from_broker()
        except Exception:
            return self._equity_from_broker()

    def _equity_from_broker(self) -> pd.DataFrame:
        """从 TimeReturn analyzer 或直接计算"""
        # 如果有 timereturn analyzer
        if hasattr(self.cerebro, '_runstrats') and self.cerebro._runstrats:
            strat = self.cerebro._runstrats[0][0]
            if hasattr(strat, 'analyzers'):
                for a in strat.analyzers:
                    if hasattr(a, 'get_analysis'):
                        try:
                            tr = a.get_analysis()
                            if hasattr(tr, 'get') and tr.get('rtot'):
                                pass
                        except Exception:
                            pass
        return pd.DataFrame()

    @staticmethod
    def compute_metrics(
        equity_series: pd.Series,
        trade_log: list[dict],
        initial_cash: float,
        final_value: float,
        risk_free_rate: float = 0.03,
    ) -> dict:
        """计算全面的绩效指标"""
        if len(equity_series) < 2:
            return {"error": "数据不足"}

        daily_returns = equity_series.pct_change().dropna()
        total_days = (equity_series.index[-1] - equity_series.index[0]).days
        years = total_days / 365.25

        # ── 收益指标 ──
        total_return = final_value / initial_cash - 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # ── 风险指标 ──
        annual_vol = daily_returns.std() * np.sqrt(242)  # A股242交易日
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()
        max_dd_end = drawdown.idxmin()
        max_dd_start = equity_series.loc[:max_dd_end].idxmax() if pd.notna(max_dd_end) else None

        # Calmar = 年化收益 / |最大回撤|
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino (下行标准差)
        downside = daily_returns[daily_returns < 0].std() * np.sqrt(242)
        sortino = (annual_return - risk_free_rate) / downside if downside > 0 else 0

        # ── 交易统计 ──
        if trade_log:
            pnls = [t["pnlcomm"] for t in trade_log]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
            avg_holding = np.mean([t["barlen"] for t in trade_log])
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_holding = 0

        return {
            "回测区间":        f"{equity_series.index[0].date()} ~ {equity_series.index[-1].date()}",
            "交易年数":        round(years, 2),
            "初始资金":        f"RMB {initial_cash:,.0f}",
            "最终资产":        f"RMB {final_value:,.0f}",
            "总收益率":        f"{total_return:.2%}",
            "年化收益率":      f"{annual_return:.2%}",
            "年化波动率":      f"{annual_vol:.2%}",
            "夏普比率":        round(sharpe, 3),
            "Sortino比率":     round(sortino, 3),
            "最大回撤":        f"{max_drawdown:.2%}",
            "回撤起始":        str(max_dd_start.date()) if max_dd_start is not None else "N/A",
            "回撤最低":        str(max_dd_end.date()) if pd.notna(max_dd_end) else "N/A",
            "Calmar比率":      round(calmar, 3),
            "总交易次数":      len(trade_log),
            "胜率":            f"{win_rate:.2%}" if trade_log else "N/A",
            "盈亏比":          f"{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A",
            "盈利因子":        round(profit_factor, 3),
            "平均持仓天数":    f"{avg_holding:.1f}" if trade_log else "N/A",
            "平均盈利":        f"RMB {avg_win:,.0f}" if wins else "N/A",
            "平均亏损":        f"RMB {avg_loss:,.0f}" if losses else "N/A",
        }

    @staticmethod
    def plot_results(
        equity_series: pd.Series,
        drawdown_series: pd.Series,
        metrics: dict,
        save_path: str = "./output/backtest_result.png",
    ):
        """绘制净值曲线和回撤图"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.suptitle("双均线交叉策略回测结果 (MA5×MA20 + 量能过滤)", fontsize=14, fontweight="bold")

        # ── 净值曲线 ──
        ax1 = axes[0]
        normalized = equity_series / equity_series.iloc[0]
        ax1.plot(normalized.index, normalized.values, color="#1976D2", linewidth=1.2, label="策略净值")
        ax1.fill_between(normalized.index, 1, normalized.values,
                        where=normalized.values >= 1, alpha=0.15, color="green")
        ax1.fill_between(normalized.index, 1, normalized.values,
                        where=normalized.values < 1, alpha=0.15, color="red")
        ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("净值 (归一化)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 标注关键指标
        info_text = (
            f"年化: {metrics.get('年化收益率', 'N/A')}  |  "
            f"夏普: {metrics.get('夏普比率', 'N/A')}  |  "
            f"最大回撤: {metrics.get('最大回撤', 'N/A')}  |  "
            f"胜率: {metrics.get('胜率', 'N/A')}"
        )
        ax1.set_title(info_text, fontsize=10, color="gray")

        # ── 回撤图 ──
        ax2 = axes[1]
        ax2.fill_between(drawdown_series.index, 0, drawdown_series.values * 100,
                        color="#E53935", alpha=0.6)
        ax2.set_ylabel("回撤 (%)")
        ax2.set_ylim(drawdown_series.min() * 100 * 1.1, 2)
        ax2.grid(True, alpha=0.3)

        # ── 月度收益热力图 (简化为柱状图) ──
        ax3 = axes[2]
        monthly = equity_series.resample("ME").last().pct_change().dropna()
        colors = ["#E53935" if x < 0 else "#43A047" for x in monthly.values]
        ax3.bar(monthly.index, monthly.values * 100, width=25, color=colors, alpha=0.7)
        ax3.set_ylabel("月收益 (%)")
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax3.grid(True, alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 图表已保存: {save_path}")


def run_backtest(
    strategy_params: dict = None,
    fromdate: str = "2018-01-01",
    todate: str = "2025-12-31",
    printlog: bool = False,
) -> dict:
    """
    执行完整回测
    Returns: {metrics: dict, equity: Series, trades: list}
    """
    print("=" * 60)
    print("  双均线交叉策略回测  MA5 × MA20 + 量能过滤")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1/4] 生成仿真行情数据...")
    data_dict = generate_all_stocks()
    print(f"  → {len(data_dict)} 只股票, 日期范围 {fromdate} ~ {todate}")

    # 2. 构建引擎
    print("[2/4] 构建回测引擎...")
    cerebro = create_cerebro(strategy_params, printlog=printlog)
    load_data_to_cerebro(cerebro, data_dict, fromdate=fromdate, todate=todate)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    # 3. 运行
    print("[3/4] 运行回测...")
    print(f"  初始资金: RMB {cerebro.broker.startingcash:,.0f}")
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    initial_cash = cerebro.broker.startingcash
    print(f"  最终资产: RMB {final_value:,.0f}")
    print(f"  总收益:   {(final_value/initial_cash - 1)*100:.2f}%")

    # 4. 分析
    print("[4/4] 计算绩效指标...")

    # 净值序列
    tr_analysis = strat.analyzers.timereturn.get_analysis()
    equity_dates = []
    equity_values = []
    cum = initial_cash
    for dt, ret in sorted(tr_analysis.items()):
        cum *= (1 + ret)
        equity_dates.append(dt)
        equity_values.append(cum)
    equity = pd.Series(equity_values, index=pd.to_datetime(equity_dates))

    # 回撤序列
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax

    # 交易记录
    trade_log = strat.trade_log

    # 计算指标
    metrics = PerformanceAnalyzer.compute_metrics(
        equity, trade_log, initial_cash, final_value
    )

    # 打印报告
    print("\n" + "─" * 50)
    print("  📋 绩效报告")
    print("─" * 50)
    for k, v in metrics.items():
        print(f"  {k:<14} {v}")
    print("─" * 50)

    # 画图
    save_path = "./output/backtest_result.png"
    PerformanceAnalyzer.plot_results(equity, drawdown, metrics, save_path)

    return {
        "metrics": metrics,
        "equity": equity,
        "drawdown": drawdown,
        "trades": trade_log,
        "cerebro": cerebro,
        "strategy": strat,
    }


# 需要导入 backtrader
import backtrader as bt


if __name__ == "__main__":
    result = run_backtest(printlog=True)
