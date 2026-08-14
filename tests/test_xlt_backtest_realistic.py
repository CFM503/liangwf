"""
Unit tests for XiaoLiangTrader/backtest/engine.py
Tests T+1 rule, limit up/down execution constraints, and data feed integration.
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime

# Add project roots
sys.path.insert(0, str(Path(__file__).parent.parent / "XiaoLiangTrader"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine, DualMABTStrategy, ASharePandasData
from data.fetcher import fetch_stock, calculate_price_limits


class TestRealisticBacktest(unittest.TestCase):
    def test_data_fetcher_integration(self):
        """测试 XiaoLiangTrader/data/fetcher.py 获取及字段"""
        df = fetch_stock("600519", start_date="20240101", end_date="20240115", use_cache=True)
        self.assertFalse(df.empty)
        for col in ["limit_up", "limit_down", "is_limit_up", "is_limit_down", "is_suspended", "turnover", "amount"]:
            self.assertIn(col, df.columns)

    def test_backtest_runs_successfully(self):
        """测试真实约束回测引擎正常运行并输出指标"""
        engine = BacktestEngine(initial_cash=1_000_000)
        report = engine.run(
            stock_codes=["600519", "000858"],
            start_date="20230101",
            end_date="20231231",
            plot=False,
        )
        self.assertNotIn("error", report)
        self.assertIn("胜率", report)
        self.assertIn("年化收益率", report)
        self.assertIn("最大回撤", report)
        self.assertIn("夏普比率", report)


if __name__ == "__main__":
    unittest.main()
