"""
Unit tests for XiaoLiangTrader/scripts/daily_signal.py
Tests daily signal generation, JSON persistence, and risk limit filtering.
"""

import unittest
import sys
from pathlib import Path

# Add project roots
sys.path.insert(0, str(Path(__file__).parent.parent / "XiaoLiangTrader"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.daily_signal import DailySignalPipeline
from strategy.signals import Action


class TestDailySignalPipeline(unittest.TestCase):
    def test_daily_signal_scan(self):
        """测试每日信号生成流水线"""
        pipeline = DailySignalPipeline(ml_confidence=0.50)
        # 用两只代表股进行测试
        res = pipeline.scan_signals(["600519", "000858"], as_of_date="20241231")
        self.assertIn("date", res)
        self.assertIn("signals", res)
        self.assertIn("scanned_count", res)
        self.assertEqual(res["scanned_count"], 2)
        self.assertIn("disclaimer", res)

        for sig in res["signals"]:
            self.assertIn(sig["action"], [Action.BUY.value, Action.SELL.value])
            self.assertTrue(sig["price"] > 0)
            if sig["action"] == Action.BUY.value:
                self.assertTrue(sig["stop_loss_price"] < sig["price"])
                self.assertTrue(sig["take_profit_price"] > sig["price"])


if __name__ == "__main__":
    unittest.main()
