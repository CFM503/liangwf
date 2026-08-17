"""
Unit tests for XiaoLiangTrader/webapp.py Streamlit App Components.
"""

import unittest
import sys
from pathlib import Path

# Add project roots
sys.path.insert(0, str(Path(__file__).parent.parent / "XiaoLiangTrader"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp import (
    load_latest_signals,
    load_live_log,
    load_affordability_data,
    load_wf_results,
)


class TestWebApp(unittest.TestCase):
    def test_load_latest_signals(self):
        """测试加载最新信号数据"""
        data = load_latest_signals()
        self.assertIsInstance(data, dict)
        if data:
            self.assertIn("date", data)
            self.assertIn("signals", data)

    def test_load_live_log(self):
        """测试加载实盘留痕日志"""
        df = load_live_log()
        self.assertIsNotNone(df)

    def test_load_affordability_data(self):
        """测试加载可买性检验数据"""
        data = load_affordability_data()
        self.assertIsInstance(data, dict)
        if data:
            self.assertIn("affordable_stocks", data)
            self.assertIn("unaffordable_stocks", data)
            self.assertEqual(data.get("total_stocks_checked"), 60)

    def test_load_wf_results(self):
        """测试加载时序滚动验证数据"""
        data = load_wf_results()
        self.assertIsInstance(data, dict)
        self.assertIn("1m_60stocks", data)
        self.assertIn("20k_capital", data)
        self.assertIn("20k_affordable", data)


if __name__ == "__main__":
    unittest.main()
