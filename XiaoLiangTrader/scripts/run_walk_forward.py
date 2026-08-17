import sys
import os
import json
import time
import argparse
from pathlib import Path

# Add XiaoLiangTrader root to sys.path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.fetcher import DEFAULT_POOL, fetch_stock
from ml_model.validator import WalkForwardValidator

RESULTS_DIR = _root / "ml_model" / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="XiaoLiangTrader 时序滚动交叉验证 (Walk-Forward) 运行工具")
    parser.add_argument("--initial-cash", type=float, default=1_000_000, help="初始回测资金 (默认 1,000,000 元)")
    parser.add_argument("--stamp-duty", type=float, default=0.0005, help="卖出印花税率 (默认 0.0005 即 0.05%)")
    parser.add_argument("--output", type=str, default=None, help="JSON 输出文件名或路径")
    parser.add_argument("--pool-filter-affordable", action="store_true", help="仅保留 2 万元本金下可买入的中低价股票池 (股价 ≤ 40元)")

    args = parser.parse_args()

    # 确定股票池
    if args.pool_filter_affordable:
        print("[Walk-Forward] 正在筛选股价 ≤ 40 元的可买中低价股票池...")
        affordable_pool = []
        for sym in DEFAULT_POOL:
            df = fetch_stock(sym, "20240101", "20260817", use_cache=True)
            if not df.empty and df["close"].iloc[-1] <= 40.0:
                affordable_pool.append(sym)
        stock_pool = affordable_pool
        default_out_name = "walk_forward_20k_affordable_only.json"
    else:
        stock_pool = DEFAULT_POOL
        default_out_name = "walk_forward_60stocks.json" if args.initial_cash >= 500_000 else "walk_forward_20k_capital.json"

    print(f"================================================================")
    print(f"🚀 开始运行时序滚动交叉验证 (Walk-Forward Validation)")
    print(f"   股票池规模: {len(stock_pool)} 只")
    print(f"   初始资金: {args.initial_cash:,.0f} 元")
    print(f"   卖出印花税: {args.stamp_duty*100:.3f}% (万分之{args.stamp_duty*10000:.1f})")
    print(f"   时间跨度: 2018-01-01 ~ 2024-12-31 (3折滚动)")
    print(f"================================================================")

    start_t = time.time()
    validator = WalkForwardValidator(
        stock_codes=stock_pool,
        forward_days=5,
        threshold=0.03,
        ml_confidence=0.55,
        initial_cash=args.initial_cash,
        stamp_duty=args.stamp_duty,
    )

    results = validator.run_walk_forward()
    validator.print_report(results)

    # 导出可复现 JSON 结果
    out_path = Path(args.output) if args.output else (RESULTS_DIR / default_out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完整时序验证结果已保存至: {out_path}")
    print(f"   总耗时: {time.time() - start_t:.1f}s")


if __name__ == "__main__":
    main()

