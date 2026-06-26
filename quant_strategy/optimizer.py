"""
参数优化模块
网格搜索 + 结果可视化，寻找最优参数组合
"""

import numpy as np
import pandas as pd
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from strategy import create_cerebro, load_data_to_cerebro
from data_generator import generate_all_stocks
from backtest_runner import PerformanceAnalyzer

import backtrader as bt


def grid_optimize(
    param_grid: dict = None,
    fromdate: str = "2018-01-01",
    todate: str = "2025-12-31",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    网格搜索参数优化

    param_grid: {
        "fast_period": [3, 5, 8],
        "slow_period": [15, 20, 30],
        "vol_mult": [1.2, 1.5, 2.0],
        "stop_loss": [-0.06, -0.08, -0.10],
        "take_profit": [0.10, 0.15, 0.20],
    }
    """
    if param_grid is None:
        param_grid = {
            "fast_period":  [3, 5, 8],
            "slow_period":  [15, 20, 30],
            "vol_mult":     [1.2, 1.5, 2.0],
            "stop_loss":    [-0.06, -0.08, -0.10],
            "take_profit":  [0.10, 0.15, 0.20],
        }

    # 生成所有组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)
    print(f"\n🔍 参数优化: {total} 种组合")
    print(f"  参数空间: {', '.join(f'{k}={v}' for k, v in param_grid.items())}")

    # 预加载数据 (只加载一次)
    print("  加载行情数据...")
    data_dict = generate_all_stocks()

    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # 跳过不合理组合 (短均线 >= 长均线)
        if params.get("fast_period", 5) >= params.get("slow_period", 20):
            continue

        try:
            cerebro = create_cerebro(params, printlog=False)
            load_data_to_cerebro(cerebro, data_dict, fromdate=fromdate, todate=todate)
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

            strat_result = cerebro.run()
            strat = strat_result[0]

            initial = cerebro.broker.startingcash
            final = cerebro.broker.getvalue()
            total_ret = final / initial - 1

            # 净值序列 → 年化
            tr = strat.analyzers.timereturn.get_analysis()
            n_days = len(tr)
            years = n_days / 242
            annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

            # 夏普
            daily_rets = [v for _, v in sorted(tr.items())]
            vol = np.std(daily_rets) * np.sqrt(242) if daily_rets else 0.001
            sharpe = (annual_ret - 0.03) / vol if vol > 0 else 0

            # 最大回撤
            dd_analysis = strat.analyzers.drawdown.get_analysis()
            max_dd = dd_analysis.get("max", {}).get("drawdown", 0) / 100

            # Calmar
            calmar = annual_ret / max_dd if max_dd > 0 else 0

            # 交易次数
            n_trades = len(strat.trade_log)
            wins = sum(1 for t in strat.trade_log if t["pnlcomm"] > 0)
            win_rate = wins / n_trades if n_trades > 0 else 0

            result = {
                **params,
                "年化收益率": round(annual_ret, 4),
                "夏普比率": round(sharpe, 3),
                "最大回撤": round(max_dd, 4),
                "Calmar": round(calmar, 3),
                "胜率": round(win_rate, 3),
                "交易次数": n_trades,
                "最终资产": round(final, 0),
            }
            results.append(result)

        except Exception as e:
            pass  # 跳过失败组合

        if (i + 1) % 20 == 0 or i + 1 == total:
            print(f"  进度: {i+1}/{total}")

    df = pd.DataFrame(results)
    if df.empty:
        print("⚠ 没有有效结果")
        return df

    # 综合评分: 0.35*夏普 + 0.25*Calmar + 0.20*胜率 + 0.20*年化 (均归一化)
    for col in ["夏普比率", "Calmar", "胜率", "年化收益率"]:
        min_v, max_v = df[col].min(), df[col].max()
        if max_v > min_v:
            df[f"_norm_{col}"] = (df[col] - min_v) / (max_v - min_v)
        else:
            df[f"_norm_{col}"] = 0.5

    df["综合评分"] = (
        0.35 * df["_norm_夏普比率"] +
        0.25 * df["_norm_Calmar"] +
        0.20 * df["_norm_胜率"] +
        0.20 * df["_norm_年化收益率"]
    )
    norm_cols = [c for c in df.columns if c.startswith("_norm_")]
    df = df.drop(columns=norm_cols)
    df = df.sort_values("综合评分", ascending=False).reset_index(drop=True)

    # 打印 TOP N
    print(f"\n{'='*80}")
    print(f"  🏆 TOP {top_n} 参数组合 (综合评分)")
    print(f"{'='*80}")
    display_cols = keys + ["年化收益率", "夏普比率", "最大回撤", "Calmar", "胜率", "交易次数", "综合评分"]
    print(df[display_cols].head(top_n).to_string(index=False))
    print(f"{'='*80}")

    # 保存
    Path("./output").mkdir(exist_ok=True)
    df.to_csv("./output/optimization_results.csv", index=False)
    print(f"\n📄 优化结果已保存: ./output/optimization_results.csv")

    # 画优化热力图
    _plot_optimization_heatmap(df, param_grid)

    return df


def _plot_optimization_heatmap(df: pd.DataFrame, param_grid: dict):
    """绘制 fast_period vs slow_period 的夏普热力图"""
    if len(df) < 4:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 热力图: fast_period vs slow_period (取其他参数最优的)
    for idx, metric in enumerate(["夏普比率", "年化收益率"]):
        ax = axes[idx]
        pivot = df.pivot_table(
            values=metric,
            index="fast_period",
            columns="slow_period",
            aggfunc="mean",
        )
        if not pivot.empty:
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("慢线周期 (slow_period)")
            ax.set_ylabel("快线周期 (fast_period)")
            ax.set_title(f"参数空间热力图 - {metric}")
            # 标注数值
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if metric == "年化收益率":
                        text = f"{val:.1%}"
                    else:
                        text = f"{val:.2f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8,
                           color="black" if 0.3 < (val - pivot.values.min()) / (pivot.values.max() - pivot.values.min() + 1e-9) < 0.7 else "white")
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("双均线策略 - 参数优化结果", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("./output/optimization_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 优化热力图已保存: ./output/optimization_heatmap.png")


if __name__ == "__main__":
    # 精简版优化 (快速验证)
    grid_optimize(
        param_grid={
            "fast_period": [3, 5, 8],
            "slow_period": [15, 20, 30],
            "vol_mult": [1.2, 1.5, 2.0],
            "stop_loss": [-0.06, -0.08, -0.10],
            "take_profit": [0.10, 0.15, 0.20],
        },
        top_n=10,
    )
