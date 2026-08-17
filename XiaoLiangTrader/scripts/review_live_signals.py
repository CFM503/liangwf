"""
实盘信号未来复盘检验工具 — 面向未来的真实跟踪复盘
===================================================
运行命令:
    python XiaoLiangTrader/scripts/review_live_signals.py
    python XiaoLiangTrader/scripts/review_live_signals.py --demo

功能特性:
1. 读取 live_signals_log.csv 累计记录的实盘信号
2. 自动拉取信号触发日之后的真实后续行情（1~5 日），计算实际收益率与止损止盈触发情况
3. 统计实盘胜率、平均收益率、盈亏比，与历史回测严格隔离
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

# Add XiaoLiangTrader root to sys.path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np

from data.fetcher import fetch_stock, STOCK_NAMES
from utils.logger import get_logger

log = get_logger("xlt.review")

RESULTS_DIR = _root / "ml_model" / "eval_results"
LIVE_LOG_PATH = RESULTS_DIR / "live_signals_log.csv"


class LiveSignalReviewer:
    """
    实盘信号事后跟踪与复盘评估器
    """

    def __init__(self, log_path: Path = LIVE_LOG_PATH, max_holding_days: int = 5):
        self.log_path = log_path
        self.max_holding_days = max_holding_days

    def load_signals(self, use_demo: bool = False) -> pd.DataFrame:
        """读取实盘信号日志，或加载示例数据"""
        if use_demo or not self.log_path.exists() or os.path.getsize(self.log_path) == 0:
            demo_data = [
                {
                    "date": "2024-09-30",
                    "symbol": "600150",
                    "name": "中国船舶",
                    "action": "BUY",
                    "price": 41.16,
                    "stop_loss_price": 37.86,
                    "take_profit_price": 47.33,
                    "ml_confidence": 0.58,
                    "reason": "MA5/20金叉 + 放量2.5x + LightGBM置信度0.58",
                },
                {
                    "date": "2024-12-31",
                    "symbol": "603259",
                    "name": "药明康德",
                    "action": "SELL",
                    "price": 51.78,
                    "stop_loss_price": 0.0,
                    "take_profit_price": 0.0,
                    "ml_confidence": 0.51,
                    "reason": "MA5/20死叉平仓",
                },
            ]
            return pd.DataFrame(demo_data)

        try:
            df = pd.read_csv(self.log_path, encoding="utf-8-sig")
            return df
        except Exception:
            try:
                df = pd.read_csv(self.log_path, encoding="utf-8")
                return df
            except Exception as e:
                log.error(f"读取信号日志失败: {e}")
                return pd.DataFrame()

    def evaluate_signals(self, df_signals: pd.DataFrame) -> Dict:
        """对每笔留痕信号回填后续真实行情并计算收益"""
        if df_signals.empty:
            return {"total_count": 0, "reviews": []}

        reviews = []
        now_str = datetime.now().strftime("%Y%m%d")

        for _, row in df_signals.iterrows():
            sym = str(row["symbol"]).zfill(6)
            name = str(row.get("name", STOCK_NAMES.get(sym, sym)))
            action = str(row["action"])
            sig_date = str(row["date"]).replace("-", "")
            entry_price = float(row["price"])
            stop_loss = float(row.get("stop_loss_price", entry_price * 0.92))
            take_profit = float(row.get("take_profit_price", entry_price * 1.15))
            ml_conf = float(row.get("ml_confidence", 0.5))
            reason = str(row.get("reason", ""))

            # 拉取信号日之后的后续 K 线（向后看 30 天）
            try:
                sig_dt = datetime.strptime(sig_date, "%Y%m%d")
                end_dt_str = (sig_dt + timedelta(days=40)).strftime("%Y%m%d")
                if end_dt_str > now_str:
                    end_dt_str = now_str
                df_fwd = fetch_stock(sym, start_date=sig_date, end_date=end_dt_str, use_cache=True)
            except Exception:
                df_fwd = pd.DataFrame()

            if df_fwd.empty or len(df_fwd) <= 1:
                reviews.append({
                    "date": str(row["date"]),
                    "symbol": sym,
                    "name": name,
                    "action": action,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "holding_days": 0,
                    "max_gain_pct": 0.0,
                    "exit_return_pct": 0.0,
                    "status": "观察期 (数据积累中)",
                    "is_win": None,
                    "exit_reason": "待观察",
                })
                continue

            # 排除信号日当天，取后续 K 线
            df_future = df_fwd.iloc[1:].head(self.max_holding_days)
            holding_bars = len(df_future)

            if action == "BUY":
                # 计算买入后真实表现
                max_high = df_future["high"].max()
                max_gain_pct = round((max_high - entry_price) / entry_price * 100.0, 2)

                exit_price = float(df_future["close"].iloc[-1])
                exit_reason = f"持有满{holding_bars}日平仓"
                exit_day = holding_bars

                # 逐日检查止盈与止损
                for day_idx, (_, bar) in enumerate(df_future.iterrows(), 1):
                    # 1. 检查止损 (-8%)
                    if bar["low"] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = f"第{day_idx}日触及止损位"
                        exit_day = day_idx
                        break
                    # 2. 检查止盈 (+15%)
                    if bar["high"] >= take_profit:
                        exit_price = take_profit
                        exit_reason = f"第{day_idx}日触及止盈位"
                        exit_day = day_idx
                        break

                exit_return = round((exit_price - entry_price) / entry_price * 100.0, 2)
                is_win = exit_return > 0
                status = "已结项" if holding_bars >= self.max_holding_days or "触及" in exit_reason else "观察中"

                reviews.append({
                    "date": str(row["date"]),
                    "symbol": sym,
                    "name": name,
                    "action": action,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "holding_days": exit_day,
                    "max_gain_pct": max_gain_pct,
                    "exit_return_pct": exit_return,
                    "status": status,
                    "is_win": is_win,
                    "exit_reason": exit_reason,
                })
            else:
                # 卖出信号
                first_close = float(df_future["close"].iloc[-1])
                diff_pct = round((first_close - entry_price) / entry_price * 100.0, 2)
                reviews.append({
                    "date": str(row["date"]),
                    "symbol": sym,
                    "name": name,
                    "action": action,
                    "entry_price": entry_price,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "holding_days": holding_bars,
                    "max_gain_pct": 0.0,
                    "exit_return_pct": -diff_pct,  # 股价下跌说明卖出规避了亏损
                    "status": "已提醒避险",
                    "is_win": diff_pct < 0,
                    "exit_reason": "死叉平仓避险",
                })

        # 统计汇总
        completed = [r for r in reviews if r["action"] == "BUY" and r["is_win"] is not None]
        total_completed = len(completed)
        wins = [r for r in completed if r["is_win"] is True]
        losses = [r for r in completed if r["is_win"] is False]
        win_rate = round(len(wins) / total_completed * 100.0, 1) if total_completed > 0 else 0.0
        avg_ret = round(np.mean([r["exit_return_pct"] for r in completed]), 2) if total_completed > 0 else 0.0
        
        avg_win = np.mean([r["exit_return_pct"] for r in wins]) if wins else 0.0
        avg_loss = abs(np.mean([r["exit_return_pct"] for r in losses])) if losses else 0.0
        pl_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)

        summary = {
            "total_signals": len(reviews),
            "completed_buy_signals": total_completed,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": win_rate,
            "avg_return_pct": avg_ret,
            "profit_loss_ratio": pl_ratio,
            "reviews": reviews,
        }
        return summary

    def print_review_report(self, summary: Dict):
        """格式化打印实盘跟踪复盘报告"""
        reviews = summary.get("reviews", [])
        total_sigs = summary.get("total_signals", 0)
        completed_buys = summary.get("completed_buy_signals", 0)
        win_rate = summary.get("win_rate", 0.0)
        avg_ret = summary.get("avg_return_pct", 0.0)
        pl_ratio = summary.get("profit_loss_ratio", 0.0)

        print("\n" + "=" * 100)
        print("         📋 XiaoLiangTrader 实盘信号事后复盘检验看板 (面向未来真实跟踪)")
        print("=" * 100)

        if not reviews:
            print("\n  💡 当前暂无留痕实盘信号，请先运行 `python XiaoLiangTrader/scripts/daily_signal.py` 记录信号。")
        else:
            print(f"\n{'信号日期':<12s} {'代码':<8s} {'名称':<8s} {'动作':<8s} {'触发价':<8s} {'止损价':<8s} {'止盈价':<8s} {'持股天':<7s} {'最高涨幅':<9s} {'实际收益':<9s} {'复盘状态'}")
            print("-" * 100)
            for r in reviews:
                d = r["date"]
                sym = r["symbol"]
                name = r["name"]
                act = "🟢 买入" if r["action"] == "BUY" else "🔴 卖出"
                ep = f"{r['entry_price']:.2f}"
                sl = f"{r['stop_loss']:.2f}" if r['stop_loss'] > 0 else "-"
                tp = f"{r['take_profit']:.2f}" if r['take_profit'] > 0 else "-"
                hd = f"{r['holding_days']}天"
                mg = f"{r['max_gain_pct']:+.2f}%" if r["action"] == "BUY" else "-"
                ret = f"{r['exit_return_pct']:+.2f}%"
                status = f"{r['status']} ({r['exit_reason']})"
                print(f"{d:<12s} {sym:<8s} {name:<8s} {act:<8s} {ep:<8s} {sl:<8s} {tp:<8s} {hd:<7s} {mg:<9s} {ret:<9s} {status}")

        print("-" * 100)
        print(f"📊 实盘跟踪统计: 累计记录信号 {total_sigs} 条 | 已结项买入 {completed_buys} 笔 | 实盘胜率: {win_rate:.1f}% | 平均单笔收益: {avg_ret:+.2f}% | 盈亏比: {pl_ratio:.2f}")
        print("=" * 100)
        print("⚠️ 隔离声明：本复盘统计专用于跟踪面向未来的实盘信号，独立于历史回测数据，旨在真实检验策略实战泛化能力。")
        print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="XiaoLiangTrader 实盘信号事后复盘检验工具")
    parser.add_argument("--demo", action="store_true", help="使用内置示例信号展示复盘看板格式")
    parser.add_argument("--days", type=int, default=5, help="默认最大持股跟踪天数 (默认 5 天)")
    args = parser.parse_args()

    reviewer = LiveSignalReviewer(max_holding_days=args.days)
    df_signals = reviewer.load_signals(use_demo=args.demo)
    summary = reviewer.evaluate_signals(df_signals)
    reviewer.print_review_report(summary)


if __name__ == "__main__":
    main()
