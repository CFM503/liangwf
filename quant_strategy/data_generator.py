"""
生成仿真A股行情数据
使用 GBM (几何布朗运动) + 跳跃扩散 模拟多只A股的OHLCV数据
特征：涨跌停限制(±10%)、T+1、成交量与价格相关性
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── 仿真参数 ──────────────────────────────────────────────
STOCK_POOL = {
    # 代码: (名称, 初始价, 年化漂移, 年化波动率)
    "600519": ("贵州茅台", 680.0, 0.12, 0.30),
    "000858": ("五粮液",   85.0, 0.10, 0.35),
    "601318": ("中国平安",  62.0, 0.05, 0.32),
    "000333": ("美的集团",  45.0, 0.08, 0.33),
    "600036": ("招商银行",  32.0, 0.07, 0.28),
    "002415": ("海康威视",  28.0, 0.09, 0.38),
    "601012": ("隆基绿能",  22.0, 0.06, 0.45),
    "300750": ("宁德时代", 120.0, 0.15, 0.42),
}

START_DATE = "2017-01-03"  # 预留一年热身，实际回测从2018开始
END_DATE   = "2025-12-31"


def _generate_single(
    code: str,
    name: str,
    init_price: float,
    mu_annual: float,
    sigma_annual: float,
    dates: pd.DatetimeIndex,
    seed: int,
) -> pd.DataFrame:
    """用 GBM + 均值回复 + 跳跃 生成单只股票日线"""
    rng = np.random.default_rng(seed)
    n = len(dates)

    dt = 1 / 242  # A股年交易日≈242
    mu_daily = mu_annual * dt
    sigma_daily = sigma_annual * np.sqrt(dt)

    # GBM 路径
    jumps = rng.choice([0, 1], size=n, p=[0.97, 0.03])  # 3%概率出现跳跃
    jump_sizes = rng.normal(0, 0.03, size=n) * jumps
    log_returns = mu_daily + sigma_daily * rng.standard_normal(n) + jump_sizes

    # 均值回复项 (防止价格偏离太远)
    log_price = np.log(init_price)
    prices = np.zeros(n)
    mean_reversion = 0.002

    for i in range(n):
        log_price += log_returns[i] - mean_reversion * (log_price - np.log(init_price))
        prices[i] = np.exp(log_price)

    close = pd.Series(prices, index=dates, name="close")

    # 涨跌停限制 (±10%)
    pct_change = close.pct_change().fillna(0).clip(-0.10, 0.10)
    close = init_price * (1 + pct_change).cumprod()

    # OHLC 生成 (保持合理关系)
    intraday_vol = np.abs(rng.normal(0.01, 0.005, n))
    high  = close * (1 + intraday_vol)
    low   = close * (1 - intraday_vol)
    open_ = close * (1 + rng.normal(0, 0.003, n))

    # 确保 high >= max(open, close) && low <= min(open, close)
    high = np.maximum(high, np.maximum(open_.values, close.values))
    low  = np.minimum(low,  np.minimum(open_.values, close.values))

    # 成交量 (与波动率正相关，与价格变化幅度正相关)
    base_vol = rng.uniform(5e6, 5e7)
    vol_noise = rng.lognormal(0, 0.5, n)
    price_change = np.abs(pct_change.values)
    volume = (base_vol * vol_noise * (1 + price_change * 10)).astype(int)
    # 量价齐升/缩量下跌 更真实的模式
    volume = np.where(pct_change.values > 0, volume * 1.3, volume * 0.8).astype(int)

    df = pd.DataFrame({
        "date":     dates,
        "open":     np.round(open_.values, 2),
        "high":     np.round(high, 2),
        "low":      np.round(low, 2),
        "close":    np.round(close.values, 2),
        "volume":   volume,
        "code":     code,
        "name":     name,
    })
    return df


def generate_all_stocks(
    stock_pool: dict = STOCK_POOL,
    start: str = START_DATE,
    end: str = END_DATE,
    save_dir: str = None,
) -> dict[str, pd.DataFrame]:
    """
    生成全部股票数据
    Returns: {code: DataFrame} 字典
    """
    # 只生成交易日 (排除周末)
    all_dates = pd.bdate_range(start, end, freq="B")
    # 模拟排除部分节假日 (简化：去掉每年春节/国庆各1周)
    masks = []
    for year in all_dates.year.unique():
        yr_dates = all_dates[all_dates.year == year]
        # 春节: 约1月底~2月初 去掉5天
        spring = yr_dates[(yr_dates.month == 1) & (yr_dates.day >= 25)].union(
                 yr_dates[(yr_dates.month == 2) & (yr_dates.day <= 5)])
        # 国庆: 10月1-7日
        national = yr_dates[(yr_dates.month == 10) & (yr_dates.day <= 7)]
        masks.append(spring)
        masks.append(national)

    all_masks = masks[0]
    for m in masks[1:]:
        all_masks = all_masks.union(m)
    trade_dates = all_dates[~all_dates.isin(all_masks)]

    result = {}
    for i, (code, (name, price, mu, sigma)) in enumerate(stock_pool.items()):
        df = _generate_single(code, name, price, mu, sigma, trade_dates, seed=42 + i)
        result[code] = df

        if save_dir:
            path = Path(save_dir)
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(path / f"{code}.csv", index=False)

    return result


if __name__ == "__main__":
    data = generate_all_stocks(save_dir="./data")
    for code, df in data.items():
        print(f"{code} {df['name'].iloc[0]}: {len(df)}条  "
              f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}  "
              f"价格 {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
