"""
数据获取模块 - 使用 akshare 获取 A 股日线数据
支持本地 CSV 缓存，避免重复请求
"""

import os
import pandas as pd
import akshare as ak

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


def fetch_stock_data(
    symbol: str,
    start_date: str = "20180101",
    end_date: str = "20251231",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取单只股票的日线数据（前复权）

    Args:
        symbol: 股票代码，如 "600519"
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        use_cache: 是否使用本地缓存

    Returns:
        DataFrame with columns: date(index), open, high, low, close, volume
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start_date}_{end_date}.csv")

    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"[缓存] {symbol} 从本地加载 {len(df)} 条记录")
        return df

    print(f"[下载] {symbol} 从 akshare 获取数据...")
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",  # 前复权
    )

    # 统一列名
    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    df.to_csv(cache_file)
    print(f"[下载] {symbol} 完成，{len(df)} 条记录已缓存")
    return df


def fetch_multi_stocks(
    symbols: list[str],
    start_date: str = "20180101",
    end_date: str = "20251231",
) -> dict[str, pd.DataFrame]:
    """批量获取多只股票数据"""
    result = {}
    for sym in symbols:
        try:
            result[sym] = fetch_stock_data(sym, start_date, end_date)
        except Exception as e:
            print(f"[错误] {sym}: {e}")
    return result


# ──────────────────────────────────────────────
# 默认标的池：各行业龙头
# ──────────────────────────────────────────────
DEFAULT_STOCKS = {
    "600519": "贵州茅台",
    "300750": "宁德时代",
    "601318": "中国平安",
    "000858": "五粮液",
    "600036": "招商银行",
}


if __name__ == "__main__":
    # 测试：获取一只股票
    df = fetch_stock_data("600519")
    print(df.tail())
    print(f"\n数据区间: {df.index[0]} ~ {df.index[-1]}")
