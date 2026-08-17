"""
Unit tests for DingTalk placeholder notifier and review_live_signals.py.
"""

import unittest
import sys
from pathlib import Path

# Add project roots
sys.path.insert(0, str(Path(__file__).parent.parent / "XiaoLiangTrader"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.dingtalk_notifier import DingTalkNotifier
from bot.notifier import BaseNotifier, Notifier
from scripts.review_live_signals import LiveSignalReviewer


class TestLiveReviewAndDingTalk(unittest.TestCase):
    def test_notifier_interface_and_dingtalk_placeholder(self):
        """测试通知接口继承与钉钉占位类"""
        self.assertTrue(issubclass(Notifier, BaseNotifier))
        self.assertTrue(issubclass(DingTalkNotifier, BaseNotifier))

        dt = DingTalkNotifier(enabled=False)
        self.assertFalse(dt.notify_report("test report"))
        self.assertFalse(dt.notify_trade("BUY", "600519", 1500.0, 100, "test"))
        self.assertFalse(dt.notify_error("test error"))
        self.assertFalse(dt.notify_kill_switch())

    def test_live_signal_reviewer_demo(self):
        """测试实盘信号复盘工具的加载与评估"""
        reviewer = LiveSignalReviewer(max_holding_days=5)
        df_demo = reviewer.load_signals(use_demo=True)
        self.assertFalse(df_demo.empty)
        self.assertIn("symbol", df_demo.columns)
        self.assertIn("action", df_demo.columns)

        summary = reviewer.evaluate_signals(df_demo)
        self.assertIn("total_signals", summary)
        self.assertIn("win_rate", summary)
        self.assertIn("reviews", summary)
        self.assertEqual(len(summary["reviews"]), len(df_demo))


if __name__ == "__main__":
    unittest.main()
