"""
Unit tests for XiaoLiangTrader/ml_model/features.py
Tests feature engineering and zero-future-function (no lookahead bias) validation.
"""

import unittest
import sys
from pathlib import Path

# Add project roots
sys.path.insert(0, str(Path(__file__).parent.parent / "XiaoLiangTrader"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from ml_model.features import (
    compute_features,
    FEATURE_COLS,
    make_forward_max_return_label,
    make_label,
)
from data.fetcher import generate_synthetic_stock


class TestFeaturesAndNoFutureLeak(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_stock("600519", "20220101", "20231231", seed=42)

    def test_feature_columns_completeness(self):
        """测试特征计算完整性与列名"""
        df_feat = compute_features(self.df)
        self.assertFalse(df_feat.empty)
        for col in FEATURE_COLS:
            self.assertIn(col, df_feat.columns, f"缺少特征列: {col}")

        # 检查特征计算后有效行数
        valid_rows = df_feat[FEATURE_COLS].dropna()
        self.assertTrue(len(valid_rows) >= len(self.df) - 65, "特征预热期超过 65 天")

    def test_label_generation_logic(self):
        """测试前瞻标签定义与开盘涨停过滤"""
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])
        mock_df = pd.DataFrame({
            "open": [10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [10.0, 10.5, 10.2, 10.1, 10.0], # Day 1 high = 10.5 (+5% vs Day 0 close 10.0)
            "low": [9.9, 9.8, 9.8, 9.8, 9.8],
            "close": [10.0, 10.2, 10.1, 10.0, 10.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
            "amount": [10000, 10000, 10000, 10000, 10000],
            "turnover": [1.0, 1.0, 1.0, 1.0, 1.0],
            "is_limit_up": [False, False, False, False, False],
        }, index=dates)

        label = make_forward_max_return_label(mock_df, forward_days=3, threshold=0.03)
        # Day 0: 未来3天最高 10.5 (+5% > 3%) -> 1
        self.assertEqual(label.iloc[0], 1)

    def test_zero_lookahead_bias_strict(self):
        """
        严禁未来函数检测（未来数据扰动测试）:
        在 T 时刻之后篡改未来行情（放大 10 倍），
        重新计算特征，断言 T 及之前的全部特征数值保持 100% 绝对一致。
        """
        df_original = self.df.copy()
        df_feat_orig = compute_features(df_original)

        split_idx = 200
        cutoff_date = df_original.index[split_idx]

        # 构造被篡改未来的数据集
        df_tampered = df_original.copy()
        # 将 split_idx 之后的收盘价、最高价、成交量全部篡改
        future_mask = df_tampered.index > cutoff_date
        df_tampered.loc[future_mask, "close"] *= 5.0
        df_tampered.loc[future_mask, "high"] *= 5.0
        df_tampered.loc[future_mask, "low"] *= 0.5
        df_tampered.loc[future_mask, "volume"] *= 10.0

        df_feat_tampered = compute_features(df_tampered)

        # 检查 T 及之前的特征矩阵
        past_orig = df_feat_orig.loc[:cutoff_date, FEATURE_COLS]
        past_tampered = df_feat_tampered.loc[:cutoff_date, FEATURE_COLS]

        # 逐列校验无差异
        for col in FEATURE_COLS:
            s_orig = past_orig[col].dropna()
            s_tamp = past_tampered[col].dropna()
            self.assertEqual(len(s_orig), len(s_tamp), f"特征 {col} 长度不匹配")
            np.testing.assert_allclose(
                s_orig.values,
                s_tamp.values,
                rtol=1e-7,
                atol=1e-7,
                err_msg=f"❌ 发现未来函数泄漏！特征 {col} 在未来数据被修改后，历史值发生了改变！"
            )


if __name__ == "__main__":
    unittest.main()
