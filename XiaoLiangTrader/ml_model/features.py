"""
特征工程 — 把 OHLCV 变成 ML 能吃的数字
=========================================
宿舍里最花时间的部分，试过几百种特征，
最后发现还是这几个最稳定。
"""

import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 OHLCV 计算技术指标特征。

    输入: df 含 open, high, low, close, volume 列
    输出: 添加了所有特征列的 df（去掉了前 N 行 NaN）

    特征列表（共约 25 个）：
    ──────────────────────────────────────────
    均线类:    ma5, ma10, ma20, ma60
    均线偏离:  ma5_bias, ma20_bias（价格偏离均线的百分比）
    动量类:    ret_1d, ret_5d, ret_10d（收益率）
    波动类:    volatility_10, volatility_20（标准差）
    成交量类:  vol_ratio_5, vol_ratio_20（量比）
    技术指标:  rsi_14, macd, macd_signal, macd_hist
    K线形态:   upper_shadow, lower_shadow, body_pct
    价格位置:  high_low_ratio, close_position
    趋势:      trend_20（20日线性回归斜率）
    """
    df = df.copy()

    # ── 均线 ──
    for p in [5, 10, 20, 60]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()

    # ── 均线偏离率（价格相对均线的偏离）──
    df["ma5_bias"] = (df["close"] - df["ma5"]) / df["ma5"]
    df["ma20_bias"] = (df["close"] - df["ma20"]) / df["ma20"]

    # ── 收益率（动量）──
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)

    # ── 波动率（标准差 / 均值）──
    df["volatility_10"] = df["ret_1d"].rolling(10).std()
    df["volatility_20"] = df["ret_1d"].rolling(20).std()

    # ── 成交量比 ──
    df["vol_ratio_5"] = df["volume"] / df["volume"].rolling(5).mean()
    df["vol_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()

    # ── RSI (14) ──
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD ──
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── K线形态特征 ──
    body = abs(df["close"] - df["open"])
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["body_pct"] = body / df["open"]

    # ── 价格位置 ──
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["open"]
    hl_range = df["high"].rolling(20).max() - df["low"].rolling(20).min()
    df["close_position"] = (df["close"] - df["low"].rolling(20).min()) / hl_range.replace(0, np.nan)

    # ── 趋势（20日线性回归斜率）──
    df["trend_20"] = df["close"].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) == 20 else 0,
        raw=False,
    )

    return df


# 所有特征列名（用于训练和预测）
FEATURE_COLS = [
    "ma5_bias", "ma20_bias",
    "ret_1d", "ret_5d", "ret_10d",
    "volatility_10", "volatility_20",
    "vol_ratio_5", "vol_ratio_20",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "upper_shadow", "lower_shadow", "body_pct",
    "high_low_ratio", "close_position",
    "trend_20",
]


def make_label(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> pd.Series:
    """
    生成标签：未来 N 天涨幅超过阈值 → 1，否则 → 0

    Args:
        df: 含 close 列的 DataFrame
        forward_days: 未来天数
        threshold: 涨幅阈值（2%）

    Returns:
        标签 Series
    """
    future_ret = df["close"].shift(-forward_days) / df["close"] - 1
    label = (future_ret > threshold).astype(int)
    return label
