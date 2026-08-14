"""
特征工程与标签定义 — 零未来函数 A 股量化特征库
==================================================
特征体系分 7 大类，共 36 个技术与量价指标：
1. 均线类:     ma5/10/20/60_bias, ma_bull_align
2. 动量类:     1/3/5/10/20 日收益率
3. 波动类:     10/20 日历史波动率, ATR(14), 布林带带宽与位置
4. 量能换手:   5/20 日量比, 10 日量价相关性, 换手率倍数, 成交额比
5. 经典指标:   RSI(6/14), MACD(DIF/DEA/柱), KDJ(K/D/J), CCI(14), WR(14)
6. K线与突破:  上下影线比, 实体比, 振幅, 20日高低点偏离, 20日突破标记
7. 趋势与连板: 20日线性回归斜率, 20日价格区间分位数, 近5日涨停频次

标签定义：
- make_forward_max_return_label: 预测未来 1~5 日内最高涨幅是否超过阈值（如 3%），
  且剔除 T+1 开盘一字涨停无法买入的假信号。
- make_forward_return_label: 预测未来 N 日收盘涨幅（向后兼容）。
"""

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 OHLCV 及量价数据计算全部技术指标特征。
    严格杜绝未来函数：所有滚动窗口与平滑计算严格基于历史及当前时刻。

    输入: df 包含 open, high, low, close, volume (可选: amount, turnover, is_limit_up)
    输出: 附加全部特征列的 df
    """
    df = df.copy()

    # 基础字段准备
    close = df["close"]
    open_p = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ════════════════════════════════════════
    # 1. 均线类 (Moving Average)
    # ════════════════════════════════════════
    for p in [5, 10, 20, 60]:
        df[f"ma{p}"] = close.rolling(p).mean()
        df[f"ma{p}_bias"] = (close - df[f"ma{p}"]) / df[f"ma{p}"]

    # 均线多头排列 (MA5 > MA10 > MA20 > MA60)
    df["ma_bull_align"] = (
        (df["ma5"] > df["ma10"]) &
        (df["ma10"] > df["ma20"]) &
        (df["ma20"] > df["ma60"])
    ).astype(int)

    # ════════════════════════════════════════
    # 2. 动量类 (Momentum)
    # ════════════════════════════════════════
    for d in [1, 3, 5, 10, 20]:
        df[f"ret_{d}d"] = close.pct_change(d)

    # ════════════════════════════════════════
    # 3. 波动类 (Volatility)
    # ════════════════════════════════════════
    daily_ret = df["ret_1d"]
    df["volatility_10"] = daily_ret.rolling(10).std()
    df["volatility_20"] = daily_ret.rolling(20).std()

    # ATR (14)
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1)),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr_14"] / close

    # 布林带 (BOLL, 20, 2)
    boll_mid = df["ma20"]
    boll_std = close.rolling(20).std()
    df["boll_upper"] = boll_mid + 2 * boll_std
    df["boll_lower"] = boll_mid - 2 * boll_std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / boll_mid.replace(0, np.nan)
    boll_range = (df["boll_upper"] - df["boll_lower"]).replace(0, np.nan)
    df["boll_position"] = (close - df["boll_lower"]) / boll_range

    # ════════════════════════════════════════
    # 4. 量能与换手率类 (Volume & Turnover)
    # ════════════════════════════════════════
    vol_ma5 = volume.rolling(5).mean().replace(0, np.nan)
    vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)
    df["vol_ratio_5"] = volume / vol_ma5
    df["vol_ratio_20"] = volume / vol_ma20
    df["vol_price_corr_10"] = close.rolling(10).corr(volume).fillna(0)

    # 换手率与成交额衍生
    if "turnover" in df.columns and (df["turnover"] > 0).any():
        turnover = df["turnover"].replace(0, np.nan).ffill().bfill().fillna(1.0)
    else:
        turnover = (volume / vol_ma20).fillna(1.0) * 1.5
    df["turnover_1d"] = turnover.fillna(1.0)
    to_ma5 = turnover.rolling(5).mean()
    to_ma20 = turnover.rolling(20).mean().replace(0, np.nan)
    df["turnover_ratio_5"] = (to_ma5 / to_ma20).fillna(1.0)

    if "amount" in df.columns:
        amount = df["amount"]
    else:
        amount = volume * close
    amt_ma5 = amount.rolling(5).mean().replace(0, np.nan)
    df["amount_ratio_5"] = amount / amt_ma5

    # ════════════════════════════════════════
    # 5. 经典技术指标 (Classic Indicators)
    # ════════════════════════════════════════
    delta = close.diff()

    # RSI (6) & RSI (14)
    for rsi_p in [6, 14]:
        gain = delta.clip(lower=0).rolling(rsi_p).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_p).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi_{rsi_p}"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # KDJ (9, 3, 3)
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = ((close - low_9) / (high_9 - low_9).replace(0, np.nan)) * 100.0
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # CCI (14)
    tp = (high + low + close) / 3.0
    tp_ma = tp.rolling(14).mean()
    mad = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).replace(0, np.nan)
    df["cci_14"] = (tp - tp_ma) / (0.015 * mad)

    # 威廉指标 WR (14)
    high_14 = high.rolling(14).max()
    low_14 = low.rolling(14).min()
    wr_range = (high_14 - low_14).replace(0, np.nan)
    df["wr_14"] = (high_14 - close) / wr_range * -100.0

    # ════════════════════════════════════════
    # 6. K线形态与突破 (Price Action & Breakout)
    # ════════════════════════════════════════
    body = abs(close - open_p)
    df["body_pct"] = body / open_p.replace(0, np.nan)
    df["upper_shadow"] = (high - df[["close", "open"]].max(axis=1)) / open_p.replace(0, np.nan)
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - low) / open_p.replace(0, np.nan)
    df["high_low_ratio"] = (high - low) / open_p.replace(0, np.nan)

    # 20日高低点偏离与突破
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    df["high_20d_bias"] = (close - high_20) / high_20.replace(0, np.nan)
    df["low_20d_bias"] = (close - low_20) / low_20.replace(0, np.nan)
    # 昨日及之前20日最高价
    prev_high_20 = high.shift(1).rolling(20).max()
    df["breakout_20d"] = (close > prev_high_20).astype(int)

    # ════════════════════════════════════════
    # 7. 趋势与连板 (Trend & Limits)
    # ════════════════════════════════════════
    # 20日线性回归斜率（纯向量化计算，避免 apply 慢速）
    x = np.arange(20)
    x_mean = 9.5
    x_var = np.sum((x - x_mean)**2)  # 66.5
    y_mean = close.rolling(20).mean()
    # sum((x - x_mean) * y)
    xy_cov = close.rolling(20).apply(lambda y_arr: np.sum((x - x_mean) * (y_arr - np.mean(y_arr))), raw=True)
    df["trend_20"] = (xy_cov / x_var) / y_mean.replace(0, np.nan)

    # 收盘价在 20 日高低点区间中的分位数 (0~1)
    hl_20_range = (high_20 - low_20).replace(0, np.nan)
    df["close_position"] = (close - low_20) / hl_20_range

    # 近5日涨停次数（如有 is_limit_up 则用标记，否则用收益率 >= 9.5% 估计）
    if "is_limit_up" in df.columns:
        is_lup = df["is_limit_up"].astype(int)
    else:
        is_lup = (df["ret_1d"] >= 0.095).astype(int)
    df["recent_limit_up_count_5d"] = is_lup.rolling(5).sum()

    return df


# ─────────────────────────────────────────────────────────────
# 统一特征列清单（36 个核心特征）
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # 均线类 (5)
    "ma5_bias", "ma10_bias", "ma20_bias", "ma60_bias", "ma_bull_align",
    # 动量类 (5)
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    # 波动类 (6)
    "volatility_10", "volatility_20", "atr_ratio", "boll_width", "boll_position", "high_low_ratio",
    # 量能换手类 (6)
    "vol_ratio_5", "vol_ratio_20", "vol_price_corr_10", "turnover_1d", "turnover_ratio_5", "amount_ratio_5",
    # 经典指标类 (9)
    "rsi_6", "rsi_14", "macd", "macd_signal", "macd_hist", "kdj_k", "kdj_d", "kdj_j", "cci_14",
    # K线与突破类 (5)
    "body_pct", "upper_shadow", "lower_shadow", "high_20d_bias", "low_20d_bias",
    # 趋势与连板类 (3)
    "breakout_20d", "trend_20", "close_position",
]


# ─────────────────────────────────────────────────────────────
# 标签生成函数
# ─────────────────────────────────────────────────────────────

def make_forward_max_return_label(
    df: pd.DataFrame,
    forward_days: int = 5,
    threshold: float = 0.03,
) -> pd.Series:
    """
    生成短线潜在机会标签（主推荐标签）：
    在 T+1 到 T+forward_days 内，未来最高价相对 T 日收盘价涨幅 >= threshold，
    且 T+1 开盘未一字涨停无法买入。

    Args:
        df: 包含 close, high, (可选 open, is_limit_up) 的 DataFrame
        forward_days: 未来考察窗口（默认 5 个交易日）
        threshold: 涨幅门槛（默认 3.0%）

    Returns:
        pd.Series: 0 或 1 标签序列
    """
    close = df["close"]
    high = df["high"]

    # 计算未来 forward_days 天的最高价 (严格使用从 T+1 开始的前向窗口)
    # rolling(forward_days).max().shift(-forward_days)
    future_max_high = high.iloc[::-1].rolling(forward_days).max().iloc[::-1].shift(-1)
    future_max_ret = (future_max_high - close) / close

    is_positive = (future_max_ret >= threshold)

    # 排除 T+1 开盘即一字涨停无法买入的情况
    if "is_limit_up" in df.columns:
        t1_limit_up = df["is_limit_up"].shift(-1).fillna(False).astype(bool)
        is_positive = is_positive & np.logical_not(t1_limit_up)

    label = is_positive.astype(int)
    # 最后 forward_days 天由于无未来完整数据设为 NaN
    label.iloc[-forward_days:] = np.nan
    return label


def make_forward_return_label(
    df: pd.DataFrame,
    forward_days: int = 3,
    threshold: float = 0.02,
) -> pd.Series:
    """
    生成未来 N 日收盘涨幅标签（向后兼容）。
    """
    future_ret = df["close"].shift(-forward_days) / df["close"] - 1.0
    label = (future_ret >= threshold).astype(int)
    label.iloc[-forward_days:] = np.nan
    return label


def make_label(
    df: pd.DataFrame,
    forward_days: int = 5,
    threshold: float = 0.03,
) -> pd.Series:
    """默认标签入口：调用短线最高涨幅机会标签"""
    return make_forward_max_return_label(df, forward_days=forward_days, threshold=threshold)
