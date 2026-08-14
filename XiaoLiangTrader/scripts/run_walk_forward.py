"""
一键复现脚本：60 只大样本池 3 折 Walk-Forward 时序滚动交叉验证
==================================================================
运行命令:
    python scripts/run_walk_forward.py
"""

import sys
import os
import json
import time
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
    print(f"================================================================")
    print(f"🚀 开始运行 60 只股票池时序滚动交叉验证 (Walk-Forward Validation)")
    print(f"   股票池规模: {len(DEFAULT_POOL)} 只跨行业核心龙头")
    print(f"   时间跨度: 2018-01-01 ~ 2024-12-31")
    print(f"================================================================")

    start_t = time.time()
    validator = WalkForwardValidator(
        stock_codes=DEFAULT_POOL,
        forward_days=5,
        threshold=0.03,
        ml_confidence=0.55,
        initial_cash=1_000_000,
    )

    results = validator.run_walk_forward()
    validator.print_report(results)

    # 导出可复现 JSON 结果
    json_path = RESULTS_DIR / "walk_forward_60stocks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完整时序验证结果已保存至: {json_path}")
    print(f"   总耗时: {time.time() - start_t:.1f}s")


if __name__ == "__main__":
    main()
