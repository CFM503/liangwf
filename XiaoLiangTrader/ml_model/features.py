"""
特征工程 — 把 OHLCV 变成 ML 能吃的数字
=========================================
2008-2010 年那会儿，宿舍里最花时间的就是调特征。
试过几百种组合，最后发现经典技术指标最稳定。
不搞花哨的，把 MACD、RSI、KDJ、布林带 这些吃透就够了。
"""

import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 OHLCV 计算全部技术指标特征。

    输入: df 含 open, high, low, close, volume 列
    输出: 添加了所有特征列的 df

    特征分 7 大类，共 30 个：
    ──────────────────────────────────────────────
    1. 均线类:     ma5, ma10, ma20, ma60, 偏离率
    2. 动量类:     1/5/10/20 日收益率
    3. 波动类:     10/20 日波动率, ATR
    4. 成交量类:   5/20 日量比
    5. 经典指标:   RSI(14), MACD, KDJ, 布林带
    6. K线形态:    上下影线, 实体比, 振幅
    7. 趋势类:     20 日线性回归斜率, 收盘价位置
    """
    df = df.copy()

    # ════════════════════════════════════════
    # 1. 均线类
    # ════════════════════════════════════════
    for p in [5, 10, 20, 60]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()

    # 偏离率（价格相对均线的偏离程度）
    df["ma5_bias"] = (df["close"] - df["ma5"]) / df["ma5"]
    df["ma20_bias"] = (df["close"] - df["ma20"]) / df["ma20"]

    # 均线多空排列（ma5 > ma10 > ma20 = 多头排列）
    df["ma_bull_align"] = ((df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])).astype(int)

    # ════════════════════════════════════════
    # 2. 动量类（收益率）
    # ════════════════════════════════════════
    for d in [1, 5, 10, 20]:
        df[f"ret_{d}d"] = df["close"].pct_change(d)

    # ════════════════════════════════════════
    # 3. 波动类
    # ════════════════════════════════════════
    daily_ret = df["close"].pct_change(1)
    df["volatility_10"] = daily_ret.rolling(10).std()
    df["volatility_20"] = daily_ret.rolling(20).std()

    # ATR (Average True Range) — 衡量波动幅度
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1)),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr_14"] / df["close"]  # ATR 占价格比例

    # ════════════════════════════════════════
    # 4. 成交量类
    # ════════════════════════════════════════
    df["vol_ratio_5"] = df["volume"] / df["volume"].rolling(5).mean()
    df["vol_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()
    # 量价关系：放量上涨 vs 放量下跌
    df["vol_price_corr_10"] = df["close"].rolling(10).corr(df["volume"])

    # ════════════════════════════════════════
    # 5. 经典指标
    # ════════════════════════════════════════

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

    # ── KDJ（随机指标）──
    # 这个是 A 股特别常用的指标
    low_9 = df["low"].rolling(9).min()
    high_9 = df["high"].rolling(9).max()
    rsv = (df["close"] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    df["kdj_k"] = rsv.ewm(com=2).mean()  # K = 2/3 * 前K + 1/3 * RSV
    df["kdj_d"] = df["kdj_k"].ewm(com=2).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # ── 布林带 (BOLL) ──
    df["boll_mid"] = df["close"].rolling(20).mean()
    boll_std = df["close"].rolling(20).std()
    df["boll_upper"] = df["boll_mid"] + 2 * boll_std
    df["boll_lower"] = df["boll_mid"] - 2 * boll_std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]
    # 价格在布林带中的位置（0=下轨, 1=上轨）
    boll_range = df["boll_upper"] - df["boll_lower"]
    df["boll_position"] = (df["close"] - df["boll_lower"]) / boll_range.replace(0, np.nan)

    # ════════════════════════════════════════
    # 6. K线形态
    # ════════════════════════════════════════
    body = abs(df["close"] - df["open"])
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["body_pct"] = body / df["open"]
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["open"]  # 振幅

    # ════════════════════════════════════════
    # 7. 趋势类
    # ════════════════════════════════════════
    # 20 日线性回归斜率（趋势方向和强度）
    df["trend_20"] = df["close"].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) == 20 else 0,
        raw=False,
    )

    # 收盘价在 20 日高低点中的位置
    hl_range = df["high"].rolling(20).max() - df["low"].rolling(20).min()
    df["close_position"] = (df["close"] - df["low"].rolling(20).min()) / hl_range.replace(0, np.nan)

    return df


# 所有特征列名（用于训练和预测）
FEATURE_COLS = [
    # 均线偏离
    "ma5_bias", "ma20_bias", "ma_bull_align",
    # 动量
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    # 波动
    "volatility_10", "volatility_20", "atr_ratio",
    # 成交量
    "vol_ratio_5", "vol_ratio_20", "vol_price_corr_10",
    # RSI
    "rsi_14",
    # MACD
    "macd", "macd_signal", "macd_hist",
    # KDJ
    "kdj_k", "kdj_d", "kdj_j",
    # 布林带
    "boll_width", "boll_position",
    # K线形态
    "body_pct", "upper_shadow", "lower_shadow", "high_low_ratio",
    # 趋势
    "trend_20", "close_position",
]


def make_label(df: pd.DataFrame, forward_days: int = 3, threshold: float = 0.02) -> pd.Series:
    """
    生成标签：未来 N 天涨幅超过阈值 → 1，否则 → 0

    Args:
        df: 含 close 列的 DataFrame
        forward_days: 未来天数（默认 3 天，比 5 天更灵敏）
        threshold: 涨幅阈值（2%）

    Returns:
        标签 Series (0 或 1)
    """
    future_ret = df["close"].shift(-forward_days) / df["close"] - 1
    label = (future_ret > threshold).astype(int)
    return label
