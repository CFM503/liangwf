"""
数据获取模块 — 用 akshare 拉取 A 股日线数据
============================================
akshare 免费无需注册，数据来源是东方财富。
首次运行需要联网，之后自动走本地 CSV 缓存。
"""

import os
import pandas as pd
import akshare as ak
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger

log = get_logger("xlt.data")

# 缓存目录（项目根目录下 data/cache/）
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 常用股票名称映射（方便日志显示）
STOCK_NAMES = {
    "600519": "贵州茅台", "300750": "宁德时代", "601318": "中国平安",
    "000858": "五粮液", "600036": "招商银行", "000001": "平安银行",
    "002594": "比亚迪", "601012": "隆基绿能", "600900": "长江电力",
}

# ──────────────────────────────────────────────
# 默认股票池 — 各行业龙头，流动性好
# 选股扫描时从这里挑，或者自己扩充
# ──────────────────────────────────────────────
DEFAULT_POOL = [
    # 白酒
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "000568",  # 泸州老窖
    # 新能源
    "300750",  # 宁德时代
    "002594",  # 比亚迪
    "601012",  # 隆基绿能
    # 金融
    "601318",  # 中国平安
    "600036",  # 招商银行
    "000001",  # 平安银行
    "601166",  # 兴业银行
    # 消费
    "000333",  # 美的集团
    "600887",  # 伊利股份
    "000651",  # 格力电器
    # 医药
    "600276",  # 恒瑞医药
    "000538",  # 云南白药
    "300760",  # 迈瑞医疗
    # 科技
    "002415",  # 海康威视
    "600900",  # 长江电力
    "601888",  # 中国中免
    "002475",  # 立讯精密
]


def fetch_stock(
    symbol: str,
    start_date: str = "20180101",
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取单只股票的日线数据（前复权）。

    Args:
        symbol: 股票代码，如 "600519"
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD，默认今天
        use_cache: 是否使用本地 CSV 缓存

    Returns:
        DataFrame，index 为 date，列: open, high, low, close, volume
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cache_file = CACHE_DIR / f"{symbol}_{start_date}_{end_date}.csv"

    # 尝试从缓存加载
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        log.info(f"[缓存] {symbol}({STOCK_NAMES.get(symbol, '')}) 加载 {len(df)} 条")
        return df

    # 从 akshare 下载
    log.info(f"[下载] {symbol} 从 akshare 获取数据...")
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",  # 前复权
        )
    except Exception as e:
        log.error(f"[下载] {symbol} 失败: {e}")
        return pd.DataFrame()

    if df.empty:
        log.warning(f"[下载] {symbol} 无数据")
        return df

    # 统一列名（akshare 返回中文列名）
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
        "成交额": "amount", "换手率": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # 写入缓存
    df.to_csv(cache_file)
    log.info(f"[下载] {symbol} 完成，{len(df)} 条已缓存")
    return df


def fetch_all_a_snapshot() -> pd.DataFrame:
    """
    获取全 A 股实时快照（一次请求拿到全部股票的实时价格、量比、涨跌幅等）。

    用于全市场选股的第一阶段快速筛选。

    Returns:
        DataFrame，列: code, name, close, pct_chg, volume, amount, amplitude,
                      turnover, pe, market_cap
    """
    log.info("[全市场] 获取 A 股实时快照...")
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        log.error(f"[全市场] 获取快照失败: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # 统一列名
    col_map = {
        "代码": "code", "名称": "name", "最新价": "close",
        "涨跌幅": "pct_chg", "成交量": "volume", "成交额": "amount",
        "振幅": "amplitude", "换手率": "turnover",
        "市盈率-动态": "pe", "总市值": "market_cap",
    }
    df = df.rename(columns=col_map)
    keep = [c for c in col_map.values() if c in df.columns]
    df = df[keep]

    # 转数值
    for col in ["close", "pct_chg", "volume", "amount", "amplitude", "turnover", "pe", "market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"[全市场] 获取到 {len(df)} 只股票")
    return df


def fetch_multi(
    symbols: list[str],
    start_date: str = "20180101",
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """批量获取多只股票数据"""
    result = {}
    for sym in symbols:
        df = fetch_stock(sym, start_date, end_date)
        if not df.empty:
            result[sym] = df
    return result


if __name__ == "__main__":
    # 测试
    df = fetch_stock("600519")
    print(df.tail())
    print(f"数据区间: {df.index[0]} ~ {df.index[-1]}")
