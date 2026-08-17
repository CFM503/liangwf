"""
小资金标的可买性检验脚本 (Check Stock Affordability)
===================================================
运行命令:
    python XiaoLiangTrader/scripts/check_affordability.py
    python XiaoLiangTrader/scripts/check_affordability.py --cash 20000 --max-single-pct 0.20

功能特性:
1. 获取股票池最新价格，计算一手（100股）建仓成本。
2. 依据初始资金与单股风控上限（如 20000 × 20% = 4000 元），统计能买与买不起的标的。
3. 将完整统计与结构化清单导出为 ml_model/eval_results/affordability_20k.json 供审计。
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add XiaoLiangTrader root to sys.path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.fetcher import DEFAULT_POOL, fetch_stock, STOCK_NAMES
from utils.logger import get_logger

log = get_logger("xlt.affordability")

RESULTS_DIR = _root / "ml_model" / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def check_affordability(
    stock_codes: List[str] = DEFAULT_POOL,
    initial_cash: float = 20000.0,
    max_single_pct: float = 0.20,
    start_date: str = "20240101",
    end_date: str = "20260817",
) -> Dict:
    """计算股票池中每只标的是否在小资金单股上限内可买入至少 1 手"""
    max_single_cash = initial_cash * max_single_pct
    max_price_threshold = max_single_cash / 100.0

    affordable = []
    unaffordable = []

    for sym in stock_codes:
        df = fetch_stock(sym, start_date=start_date, end_date=end_date, use_cache=True)
        if df.empty:
            continue
        price = float(df["close"].iloc[-1])
        lot_cost = round(price * 100.0, 2)
        name = STOCK_NAMES.get(sym, sym)

        item = {
            "symbol": sym,
            "name": name,
            "latest_price": round(price, 2),
            "one_lot_cost": lot_cost,
            "max_single_cash_limit": max_single_cash,
        }

        if price <= max_price_threshold:
            item["status"] = "AFFORDABLE"
            affordable.append(item)
        else:
            item["status"] = "UNAFFORDABLE"
            item["shortfall_per_lot"] = round(lot_cost - max_single_cash, 2)
            unaffordable.append(item)

    # 排序
    affordable.sort(key=lambda x: x["latest_price"])
    unaffordable.sort(key=lambda x: x["latest_price"], reverse=True)

    total_count = len(affordable) + len(unaffordable)
    aff_ratio = round(len(affordable) / total_count * 100.0, 1) if total_count > 0 else 0.0
    unaff_ratio = round(len(unaffordable) / total_count * 100.0, 1) if total_count > 0 else 0.0

    result = {
        "initial_cash": initial_cash,
        "max_single_pct": max_single_pct,
        "max_single_cash_limit": max_single_cash,
        "max_price_threshold": max_price_threshold,
        "total_stocks_checked": total_count,
        "affordable_count": len(affordable),
        "affordable_ratio": f"{aff_ratio}%",
        "unaffordable_count": len(unaffordable),
        "unaffordable_ratio": f"{unaff_ratio}%",
        "affordable_stocks": affordable,
        "unaffordable_stocks": unaffordable,
    }
    return result


def print_report(res: Dict):
    """终端打印可买性报告"""
    print("\n" + "=" * 80)
    print("         📊 XiaoLiangTrader 小资金标的可买性检验报告")
    print("=" * 80)
    print(f"  初始本金: {res['initial_cash']:,.0f} 元 | 单股仓位上限: {res['max_single_pct']*100:.0f}% (最多 {res['max_single_cash_limit']:,.0f} 元/股)")
    print(f"  买入一手 (100股) 最高允许股价: ≤ {res['max_price_threshold']:.2f} 元")
    print(f"  总检验标的: {res['total_stocks_checked']} 只 | 可买: {res['affordable_count']} 只 ({res['affordable_ratio']}) | 买不起: {res['unaffordable_count']} 只 ({res['unaffordable_ratio']})")
    print("-" * 80)

    print(f"\n【🟢 可买股票清单 (共 {res['affordable_count']} 只)】:")
    for s in res["affordable_stocks"]:
        print(f"  {s['symbol']} {s['name']:<8s} 现价: {s['latest_price']:>6.2f} 元 | 1手成本: {s['one_lot_cost']:>7.0f} 元")

    print(f"\n【🔴 买不起股票清单 (一手成本 > {res['max_single_cash_limit']:,.0f} 元，共 {res['unaffordable_count']} 只)】:")
    for s in res["unaffordable_stocks"]:
        print(f"  {s['symbol']} {s['name']:<8s} 现价: {s['latest_price']:>7.2f} 元 | 1手成本: {s['one_lot_cost']:>8.0f} 元 (超额: +{s['shortfall_per_lot']:>7.0f} 元)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="XiaoLiangTrader 小资金标的可买性检验工具")
    parser.add_argument("--cash", type=float, default=20000.0, help="初始资金 (默认 20000 元)")
    parser.add_argument("--max-single-pct", type=float, default=0.20, help="单股最大仓位比例 (默认 0.20)")
    parser.add_argument("--output", type=str, default=None, help="导出 JSON 路径")
    args = parser.parse_args()

    res = check_affordability(
        stock_codes=DEFAULT_POOL,
        initial_cash=args.cash,
        max_single_pct=args.max_single_pct,
    )
    print_report(res)

    out_file = Path(args.output) if args.output else (RESULTS_DIR / "affordability_20k.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    log.info(f"[可买性检验] 结果已成功导出至: {out_file}")


if __name__ == "__main__":
    main()
