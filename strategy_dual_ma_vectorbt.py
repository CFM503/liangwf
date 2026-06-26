"""
双均线交叉策略 - VectorBT 实现
================================
向量化回测，速度快，适合大规模参数扫描
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib

matplotlib.use("Agg")

from data_fetcher import fetch_stock_data, DEFAULT_STOCKS


# ═══════════════════════════════════════════════
# 1. 信号生成
# ═══════════════════════════════════════════════
def generate_signals(
    close: pd.Series,
    volume: pd.Series,
    fast_period: int = 5,
    slow_period: int = 20,
    vol_period: int = 20,
    vol_mult: float = 1.5,
) -> tuple[pd.Series, pd.Series]:
    """
    生成买入/卖出信号

    Returns:
        entries (bool Series): 买入信号
        exits (bool Series): 卖出信号
    """
    fast_ma = close.rolling(fast_period).mean()
    slow_ma = close.rolling(slow_period).mean()
    vol_ma = volume.rolling(vol_period).mean()

    # 金叉：fast_ma 上穿 slow_ma
    golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    # 死叉：fast_ma 下穿 slow_ma
    death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    # 成交量过滤
    vol_filter = volume > vol_ma * vol_mult

    entries = golden_cross & vol_filter
    exits = death_cross

    return entries, exits


# ═══════════════════════════════════════════════
# 2. 单标的回测
# ═══════════════════════════════════════════════
def run_single_backtest(
    symbol: str,
    start_date: str = "20180101",
    end_date: str = "20251231",
    fast_period: int = 5,
    slow_period: int = 20,
    vol_mult: float = 1.5,
    init_cash: float = 200_000,  # 单股分配20万（总100万的20%）
    stop_loss: float = 0.08,
    take_profit: float = 0.15,
) -> vbt.Portfolio:
    """对单只股票运行回测，返回 Portfolio 对象"""

    df = fetch_stock_data(symbol, start_date, end_date)
    entries, exits = generate_signals(
        df["close"], df["volume"],
        fast_period=fast_period,
        slow_period=slow_period,
        vol_mult=vol_mult,
    )

    # 跟踪止盈：VectorBT 没有内置 trailing stop，
    # 用固定止损 + 止盈模拟，跟踪止盈在 exits 信号中已覆盖（死叉时卖出）
    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries,
        exits=~entries & exits,  # 确保只在 exit 信号时卖出
        init_cash=init_cash,
        fees=0.00025,           # 佣金万2.5
        slippage=0.001,         # 滑点0.1%
        sl_stop=stop_loss,      # 止损
        tp_stop=take_profit,    # 止盈
        freq="1D",
    )

    return pf


# ═══════════════════════════════════════════════
# 3. 多标的组合回测
# ═══════════════════════════════════════════════
def run_multi_backtest(
    stock_codes: list[str] | None = None,
    start_date: str = "20180101",
    end_date: str = "20251231",
    initial_cash: float = 1_000_000,
    fast_period: int = 5,
    slow_period: int = 20,
    vol_mult: float = 1.5,
    plot: bool = True,
) -> dict:
    """多标的组合回测"""
    if stock_codes is None:
        stock_codes = list(DEFAULT_STOCKS.keys())[:3]

    # 每只股票分配资金（均等分配，单股不超过20%）
    per_stock_cash = initial_cash * 0.20

    all_stats = []
    total_final_value = 0

    print(f"{'='*60}")
    print(f"VectorBT 回测 | 初始资金: {initial_cash:,.0f}")
    print(f"标的: {', '.join(stock_codes)}")
    print(f"参数: MA{fast_period}/{slow_period}, 量比>{vol_mult}x")
    print(f"{'='*60}")

    for code in stock_codes:
        try:
            pf = run_single_backtest(
                code, start_date, end_date,
                fast_period=fast_period,
                slow_period=slow_period,
                vol_mult=vol_mult,
                init_cash=per_stock_cash,
            )

            stats = pf.stats()
            total_final_value += pf.final_value()

            print(f"\n── {code} ({DEFAULT_STOCKS.get(code, '')}) ──")
            print(f"  最终净值: {pf.final_value():,.0f}")
            print(f"  总收益率: {pf.total_return() * 100:.2f}%")
            print(f"  最大回撤: {pf.max_drawdown() * 100:.2f}%")
            print(f"  夏普比率: {pf.sharpe_ratio():.2f}")
            print(f"  交易次数: {pf.trades.count()}")

            all_stats.append({
                "stock": code,
                "name": DEFAULT_STOCKS.get(code, ""),
                "final_value": pf.final_value(),
                "total_return": pf.total_return() * 100,
                "max_drawdown": pf.max_drawdown() * 100,
                "sharpe": pf.sharpe_ratio(),
                "trades": pf.trades.count(),
                "win_rate": pf.trades.win_rate() * 100 if pf.trades.count() > 0 else 0,
            })

            if plot:
                fig = pf.plot()
                fig.write_image(f"vbt_{code}.png", scale=2)
                print(f"  图表: vbt_{code}.png")

        except Exception as e:
            print(f"[错误] {code}: {e}")

    # 汇总
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        total_return = (total_final_value / initial_cash - 1) * 100

        print(f"\n{'='*60}")
        print("组合汇总")
        print(f"{'='*60}")
        print(f"  初始资金:  {initial_cash:,.0f}")
        print(f"  最终净值:  {total_final_value:,.0f}")
        print(f"  总收益率:  {total_return:.2f}%")
        print(f"  平均夏普:  {stats_df['sharpe'].mean():.2f}")
        print(f"  平均胜率:  {stats_df['win_rate'].mean():.1f}%")
        print(f"  最大单股回撤: {stats_df['max_drawdown'].max():.2f}%")
        print(f"{'='*60}")

        print("\n个股明细:")
        print(stats_df.to_string(index=False))

        return {
            "total_return": total_return,
            "final_value": total_final_value,
            "avg_sharpe": stats_df["sharpe"].mean(),
            "avg_win_rate": stats_df["win_rate"].mean(),
            "max_drawdown": stats_df["max_drawdown"].max(),
            "details": stats_df,
        }

    return {}


# ═══════════════════════════════════════════════
# 4. 参数网格优化
# ═══════════════════════════════════════════════
def parameter_optimization(
    symbol: str = "600519",
    start_date: str = "20180101",
    end_date: str = "20251231",
    plot: bool = True,
):
    """
    VectorBT 参数网格优化
    利用向量化优势，快速扫描大量参数组合
    """
    df = fetch_stock_data(symbol, start_date, end_date)

    # 参数范围
    fast_range = np.arange(3, 11)          # MA3 ~ MA10
    slow_range = np.arange(15, 35, 5)      # MA15 ~ MA30
    vol_range = np.array([1.0, 1.3, 1.5, 1.8, 2.0])

    # 组合数
    combos = 0
    for f in fast_range:
        for s in slow_range:
            if f < s:
                combos += 1
    total = combos * len(vol_range)
    print(f"参数组合: {total} 个")

    best_sharpe = -999
    best_params = None
    best_pf = None
    results = []

    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            for vol_m in vol_range:
                try:
                    pf = run_single_backtest(
                        symbol, start_date, end_date,
                        fast_period=int(fast),
                        slow_period=int(slow),
                        vol_mult=float(vol_m),
                    )
                    sharpe = pf.sharpe_ratio()
                    if np.isnan(sharpe):
                        sharpe = 0

                    results.append({
                        "fast": int(fast),
                        "slow": int(slow),
                        "vol_mult": float(vol_m),
                        "sharpe": sharpe,
                        "total_return": pf.total_return() * 100,
                        "max_drawdown": pf.max_drawdown() * 100,
                        "trades": pf.trades.count(),
                    })

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = (int(fast), int(slow), float(vol_m))
                        best_pf = pf
                except Exception:
                    pass

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)

    print(f"\n{'='*60}")
    print(f"参数优化结果 | 标的: {symbol} ({DEFAULT_STOCKS.get(symbol, '')})")
    print(f"{'='*60}")
    print("\nTOP 10 参数组合:")
    print(results_df.head(10).to_string(index=False))

    if best_params:
        print(f"\n最优: MA{best_params[0]}/{best_params[1]}, "
              f"量比>{best_params[2]}x")
        print(f"  夏普: {best_sharpe:.2f}")
        print(f"  收益: {results_df.iloc[0]['total_return']:.2f}%")
        print(f"  回撤: {results_df.iloc[0]['max_drawdown']:.2f}%")

    # 热力图
    if plot and not results_df.empty:
        _plot_heatmap(results_df, symbol)

    return results_df


def _plot_heatmap(df: pd.DataFrame, symbol: str):
    """绘制参数热力图"""
    try:
        import matplotlib.pyplot as plt

        # 取 vol_mult=1.5 的切面
        subset = df[df["vol_mult"] == 1.5].copy()
        if subset.empty:
            subset = df.copy()

        pivot = subset.pivot_table(
            values="sharpe", index="slow", columns="fast", aggfunc="max"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Fast MA Period")
        ax.set_ylabel("Slow MA Period")
        ax.set_title(f"Sharpe Ratio Heatmap | {symbol} (vol_mult=1.5)")

        # 标注数值
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, label="Sharpe Ratio")
        plt.tight_layout()
        plt.savefig("param_heatmap.png", dpi=150)
        print("\n热力图已保存: param_heatmap.png")
    except Exception as e:
        print(f"热力图绘制失败: {e}")


# ═══════════════════════════════════════════════
# 5. 入口
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "optimize":
        parameter_optimization()
    elif mode == "single":
        code = sys.argv[2] if len(sys.argv) > 2 else "600519"
        pf = run_single_backtest(code)
        print(pf.stats())
    else:
        run_multi_backtest()
