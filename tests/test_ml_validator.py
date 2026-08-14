"""
Unit tests for XiaoLiangTrader/ml_model/validator.py
Tests expanding window walk-forward validation logic.
"""

import unittest
import sys
from pathlib import Path

# Add project roots
sys.path.insert(0, str(Path(__file__).parent.parent / "XiaoLiangTrader"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_model.validator import WalkForwardValidator


class TestWalkForwardValidator(unittest.TestCase):
    def test_walk_forward_execution(self):
        """测试简版 Walk-Forward 滚动切分与指标结构"""
        validator = WalkForwardValidator(
            stock_codes=["600519", "000858"],
            forward_days=3,
            threshold=0.02,
            ml_confidence=0.55,
        )
        custom_folds = [
            {
                "name": "Test Fold 2023",
                "train_start": "20210101",
                "train_end": "20221231",
                "test_start": "20230101",
                "test_end": "20230630",
            }
        ]
        results = validator.run_walk_forward(custom_folds=custom_folds)
        self.assertIn("folds", results)
        self.assertEqual(len(results["folds"]), 1)

        f = results["folds"][0]
        self.assertIn("classification", f)
        self.assertIn("backtest", f)
        self.assertIn("lightgbm", f["classification"])
        self.assertIn("baseline", f["backtest"])
        self.assertIn("win_rate", f["backtest"]["baseline"])


if __name__ == "__main__":
    unittest.main()
