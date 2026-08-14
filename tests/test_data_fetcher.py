"""
Unit tests for data_fetcher.py
Tests field completeness, limit up/down rules, suspension flags, and data caching.
"""

import unittest
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from data_fetcher import (
    fetch_stock_data,
    fetch_multi_stocks,
    calculate_price_limits,
    generate_synthetic_stock_data,
    DEFAULT_STOCKS,
)


class TestDataFetcher(unittest.TestCase):
    def test_synthetic_data_generation(self):
        """测试合成数据生成及字段完整性"""
        df = generate_synthetic_stock_data("600519", start_date="20230101", end_date="20231231")
        self.assertFalse(df.empty)
        expected_cols = [
            "open", "high", "low", "close", "volume", "amount",
            "turnover", "pct_chg", "limit_up", "limit_down",
            "is_limit_up", "is_limit_down", "is_suspended", "is_st"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"缺少必要字段: {col}")
        
        # 验证价格合理性
        self.assertTrue((df["high"] >= df["low"]).all())
        self.assertTrue((df["high"] >= df["close"]).all())
        self.assertTrue((df["high"] >= df["open"]).all())
        self.assertTrue((df["low"] <= df["close"]).all())
        self.assertTrue((df["low"] <= df["open"]).all())

    def test_limit_up_down_calculation_mainboard(self):
        """测试主板 10% 涨跌停计算"""
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        mock_df = pd.DataFrame({
            "open": [100.0, 110.0, 99.0],
            "high": [110.0, 110.0, 100.0],
            "low": [99.0, 108.0, 99.0],
            "close": [100.0, 110.0, 99.0],
            "volume": [10000, 12000, 8000],
            "amount": [1000000, 1320000, 792000],
            "turnover": [1.0, 1.2, 0.8],
        }, index=dates)

        res = calculate_price_limits(mock_df, symbol="600519", is_st=False)
        # 第二天昨收 100.0，涨停应为 110.0，跌停应为 90.0
        self.assertEqual(float(res.loc[dates[1], "limit_up"]), 110.0)
        self.assertEqual(float(res.loc[dates[1], "limit_down"]), 90.0)
        self.assertTrue(bool(res.loc[dates[1], "is_limit_up"]))
        self.assertFalse(bool(res.loc[dates[1], "is_limit_down"]))

    def test_limit_up_down_calculation_chinext(self):
        """测试创业板 2020-08-24 之后 20% 涨跌停计算"""
        dates = pd.to_datetime(["2020-08-21", "2020-08-24", "2020-08-25"])
        mock_df = pd.DataFrame({
            "open": [10.0, 12.0, 14.4],
            "high": [10.0, 12.0, 14.4],
            "low": [10.0, 10.0, 12.0],
            "close": [10.0, 12.0, 14.4],
            "volume": [1000, 1000, 1000],
            "amount": [10000, 12000, 14400],
            "turnover": [1.0, 1.0, 1.0],
        }, index=dates)

        res = calculate_price_limits(mock_df, symbol="300750", is_st=False)
        # 2020-08-24 当日及之后创业板涨停为 20%
        # 2020-08-24 前收 10.0 -> 涨停 12.0
        self.assertEqual(float(res.loc[dates[1], "limit_up"]), 12.0)
        # 2020-08-25 前收 12.0 -> 涨停 14.4 (12 * 1.2)
        self.assertEqual(float(res.loc[dates[2], "limit_up"]), 14.4)

    def test_limit_up_down_calculation_st(self):
        """测试 ST 股票 5% 涨跌停计算"""
        dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
        mock_df = pd.DataFrame({
            "open": [10.0, 10.5],
            "high": [10.0, 10.5],
            "low": [10.0, 10.0],
            "close": [10.0, 10.5],
            "volume": [1000, 1000],
            "amount": [10000, 10500],
            "turnover": [1.0, 1.0],
        }, index=dates)

        res = calculate_price_limits(mock_df, symbol="600000", is_st=True)
        # 前收 10.0 -> ST涨停 10.5, 跌停 9.5
        self.assertEqual(float(res.loc[dates[1], "limit_up"]), 10.5)
        self.assertEqual(float(res.loc[dates[1], "limit_down"]), 9.5)
        self.assertTrue(bool(res.loc[dates[1], "is_limit_up"]))

    def test_suspension_detection(self):
        """测试停牌识别"""
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        mock_df = pd.DataFrame({
            "open": [10.0, 10.0, 10.2],
            "high": [10.0, 10.0, 10.5],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.2],
            "volume": [1000, 0, 1500],
            "amount": [10000, 0, 15300],
            "turnover": [1.0, 0.0, 1.5],
        }, index=dates)

        res = calculate_price_limits(mock_df, symbol="600519")
        self.assertFalse(bool(res.loc[dates[0], "is_suspended"]))
        self.assertTrue(bool(res.loc[dates[1], "is_suspended"]))
        self.assertFalse(bool(res.loc[dates[2], "is_suspended"]))

    def test_fetch_stock_data_fallback(self):
        """测试真实/备用源数据获取"""
        df = fetch_stock_data("600519", start_date="20240101", end_date="20240115", use_cache=False)
        self.assertFalse(df.empty)
        self.assertTrue(len(df) >= 5)
        self.assertIn("limit_up", df.columns)
        self.assertIn("is_limit_up", df.columns)
        self.assertIn("turnover", df.columns)


if __name__ == "__main__":
    unittest.main()
