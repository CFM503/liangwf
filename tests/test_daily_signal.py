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

    def test_notification_formatting(self):
        """测试通知文本排版与格式化"""
        pipeline = DailySignalPipeline()
        mock_payload = {
            "date": "2026-08-14",
            "scanned_count": 60,
            "buy_count": 1,
            "sell_count": 1,
            "no_signal_count": 58,
            "signals": [
                {
                    "symbol": "600150",
                    "name": "中国船舶",
                    "action": "BUY",
                    "action_label": "🟢 建议买入",
                    "price": 41.16,
                    "stop_loss_price": 37.86,
                    "take_profit_price": 47.33,
                    "ml_confidence": 0.58,
                    "reason": "MA5/20金叉",
                }
            ],
            "disclaimer": "仅供参考",
        }
        text = pipeline.format_notification_text(mock_payload)
        self.assertIn("600150", text)
        self.assertIn("中国船舶", text)
        self.assertIn("41.16", text)
        self.assertIn("扫描 60 只", text)


if __name__ == "__main__":
    unittest.main()
