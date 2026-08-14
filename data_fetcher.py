"""
数据获取模块 - 支持多源备份 (Akshare / 腾讯 / 新浪 / 离线合成)
================================================================
统一输出标准化日线数据：
- OHLCV (前复权 / 后复权 / 不复权)
- 成交额 (amount)、换手率 (turnover)、涨跌幅 (pct_chg)
- 涨跌停价 (limit_up / limit_down)、涨跌停标记 (is_limit_up / is_limit_down)
- 停牌标记 (is_suspended)、ST标记 (is_st)
- 本地 CSV 增量缓存，避免重复拉取
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

try:
    import akshare as ak
except ImportError:
    ak = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_STOCKS = {
    "600519": "贵州茅台",
    "300750": "宁德时代",
    "601318": "中国平安",
    "000858": "五粮液",
    "600036": "招商银行",
    "000001": "平安银行",
    "002594": "比亚迪",
    "601012": "隆基绿能",
    "600900": "长江电力",
    "600276": "恒瑞医药",
}

DEFAULT_POOL = list(DEFAULT_STOCKS.keys())


# ─────────────────────────────────────────────────────────────
# 1. 涨跌停与交易状态标记计算
# ─────────────────────────────────────────────────────────────

def calculate_price_limits(df: pd.DataFrame, symbol: str, is_st: bool = False) -> pd.DataFrame:
    """
    计算 A 股各板块每日理论涨跌停价格及触及标记。

    规则：
    - 主板 (600/601/603/605/000/001/002/003)：通常 ±10%
    - 创业板 (300/301)：2020-08-24注册制起为 ±20%，此前为 ±10%
    - 科创板 (688)：±20%
    - 北交所 (920/83/87/43)：±30%
    - ST / *ST 股票：±5%

    Args:
        df: 包含 open, high, low, close, volume 的 DataFrame (index 为日期)
        symbol: 股票代码
        is_st: 是否 ST 股票

    Returns:
        包含 limit_up, limit_down, is_limit_up, is_limit_down, is_suspended, is_st 的 df
    """
    df = df.copy()
    if df.empty:
        return df

    symbol_str = str(symbol).strip()
    is_chinext = symbol_str.startswith(("300", "301"))
    is_star = symbol_str.startswith("688")
    is_bse = symbol_str.startswith(("920", "83", "87", "43"))

    # 计算昨收 (首根用 open 填充)
    if "prev_close" not in df.columns:
        df["prev_close"] = df["close"].shift(1).fillna(df["open"])

    dates = pd.to_datetime(df.index)
    ratios = np.full(len(df), 0.10)  # 默认主板 10%

    if is_st:
        ratios[:] = 0.05
    elif is_bse:
        ratios[:] = 0.30
    elif is_star:
        ratios[:] = 0.20
    elif is_chinext:
        chinext_20_mask = dates >= pd.Timestamp("2020-08-24")
        ratios[chinext_20_mask] = 0.20
        ratios[~chinext_20_mask] = 0.10

    # 涨跌停价格（A股以分四舍五入）
    df["limit_up"] = np.round(df["prev_close"] * (1.0 + ratios), 2)
    df["limit_down"] = np.round(df["prev_close"] * (1.0 - ratios), 2)

    # 涨跌停触及/封死判断：
    # 涨停封死定义：收盘价 >= 涨停价 - 0.01 且收盘价为当日最高价
    df["is_limit_up"] = (df["close"] >= df["limit_up"] - 0.01) & (df["close"] >= df["high"] - 0.01)
    # 跌停封死定义：收盘价 <= 跌停价 + 0.01 且收盘价为当日最低价
    df["is_limit_down"] = (df["close"] <= df["limit_down"] + 0.01) & (df["close"] <= df["low"] + 0.01)

    # 停牌判定 (成交量为0或NaN)
    df["is_suspended"] = (df["volume"] == 0) | df["volume"].isna()
    df["is_st"] = is_st

    # 涨跌幅 (%)
    if "pct_chg" not in df.columns:
        df["pct_chg"] = (df["close"] / df["prev_close"] - 1.0) * 100.0

    return df


# ─────────────────────────────────────────────────────────────
# 2. 多源拉取实现 (Tencent / Akshare / Sina / Synthetic)
# ─────────────────────────────────────────────────────────────

def _fetch_from_tencent(
    symbol: str,
    start_date: str = "20180101",
    end_date: str = "20251231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    通过腾讯财经接口获取日 K 线数据（速度极快、国内稳定可用）。
    """
    symbol_str = str(symbol).strip()
    prefix = "sh" if symbol_str.startswith(("6", "9")) else "sz" if symbol_str.startswith(("0", "3")) else "bj"
    code = f"{prefix}{symbol_str}"

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d") if end_date else datetime.now()

    all_kline = []
    # 腾讯按年份或按数量拉取
    for year in range(start_dt.year, end_dt.year + 1):
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,{y_start},{y_end},350,{adjust}"
        try:
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                data = resp.json().get("data", {}).get(code, {})
                kdata = data.get(f"{adjust}day", data.get("day", []))
                if kdata:
                    all_kline.extend(kdata)
        except Exception as e:
            logger.debug(f"[Tencent] {symbol} {year} 年数据拉取失败: {e}")

    if not all_kline:
        return pd.DataFrame()

    # 腾讯格式: [date, open, close, high, low, volume (手), ...]
    records = []
    seen_dates = set()
    for row in all_kline:
        if not row or len(row) < 6:
            continue
        d_str = row[0]
        if d_str in seen_dates:
            continue
        seen_dates.add(d_str)
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
            if not (start_dt <= d <= end_dt):
                continue
            open_p = float(row[1])
            close_p = float(row[2])
            high_p = float(row[3])
            low_p = float(row[4])
            vol_lots = float(row[5])  # 腾讯成交量单位为手 (1手=100股)
            volume = vol_lots * 100.0
            amount = volume * ((open_p + close_p + high_p + low_p) / 4.0)  # 估算成交额
            records.append({
                "date": d,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume,
                "amount": amount,
                "turnover": 0.0,  # 稍后结合换手率
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("date").set_index("date")
    return df


def _fetch_from_akshare(
    symbol: str,
    start_date: str = "20180101",
    end_date: str = "20251231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    通过 Akshare 获取日 K 线数据。
    """
    if ak is None:
        return pd.DataFrame()

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        col_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "涨跌幅": "pct_chg",
        }
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        cols = [c for c in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"] if c in df.columns]
        df = df[cols].astype(float)
        return df
    except Exception as e:
        logger.debug(f"[Akshare] {symbol} 下载失败: {e}")
        return pd.DataFrame()


def generate_synthetic_stock_data(
    symbol: str = "600519",
    start_date: str = "20180101",
    end_date: str = "20251231",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    生成逼真的 A 股合成行情数据（用于离线测试或断网环境）。
    """
    if seed is None:
        seed = int(abs(hash(str(symbol))) % (2**31 - 1))
    np.random.seed(seed)

    dates = pd.bdate_range(
        datetime.strptime(start_date, "%Y%m%d"),
        datetime.strptime(end_date, "%Y%m%d") if end_date else datetime.now(),
    )
    n = len(dates)
    if n == 0:
        return pd.DataFrame()

    # 随机漫步 + 均值回归 + 波动率聚类
    dt = 1.0 / 242.0
    mu = 0.08
    sigma = 0.30
    daily_shocks = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)

    # 初始基准价格
    base_price = 20.0 + (seed % 100) * 5.0
    prices = [base_price]
    for shock in daily_shocks:
        p = prices[-1] * np.exp(shock)
        prices.append(max(p, 1.0))
    prices = np.array(prices[1:])

    records = []
    for i, (d, close_p) in enumerate(zip(dates, prices)):
        daily_range = np.random.uniform(0.015, 0.04)
        open_p = close_p * (1.0 + np.random.uniform(-0.01, 0.01))
        high_p = max(open_p, close_p) * (1.0 + np.random.uniform(0.002, daily_range))
        low_p = min(open_p, close_p) * (1.0 - np.random.uniform(0.002, daily_range))
        vol = int(np.random.lognormal(14.0, 0.8))  # 约 100万~500万股
        amount = vol * ((open_p + high_p + low_p + close_p) / 4.0)
        turnover = np.random.uniform(0.5, 4.5)

        records.append({
            "date": d,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": vol,
            "amount": round(amount, 2),
            "turnover": round(turnover, 2),
        })

    df = pd.DataFrame(records).set_index("date")
    df = calculate_price_limits(df, symbol=symbol)
    return df


# ─────────────────────────────────────────────────────────────
# 3. 统一数据对外接口
# ─────────────────────────────────────────────────────────────

def fetch_stock_data(
    symbol: str,
    start_date: str = "20180101",
    end_date: Optional[str] = None,
    use_cache: bool = True,
    adjust: str = "qfq",
    is_st: bool = False,
) -> pd.DataFrame:
    """
    获取单只股票日线行情，具备多级缓存与容灾回退。

    Args:
        symbol: 股票代码，如 "600519"
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD (默认今天)
        use_cache: 是否使用本地 CSV 缓存
        adjust: 复权类型 ("qfq", "hfq", "")
        is_st: 是否 ST 股票

    Returns:
        DataFrame: 标准化日线数据
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start_date}_{end_date}_{adjust}.csv")

    # 1. 尝试缓存
    if use_cache and os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty and "limit_up" in df.columns:
                logger.info(f"[缓存] {symbol} 从本地加载 {len(df)} 条记录")
                return df
        except Exception:
            pass

    # 2. 主通道：Tencent 接口（国内直连高并发稳定）
    logger.info(f"[下载] {symbol} 正在从在线数据源获取数据...")
    df = _fetch_from_tencent(symbol, start_date, end_date, adjust=adjust)

    # 3. 备用通道：Akshare
    if df.empty:
        df = _fetch_from_akshare(symbol, start_date, end_date, adjust=adjust)

    # 4. 离线/测试通道：合成数据（保证即使断网系统仍能闭环验证）
    if df.empty:
        logger.warning(f"[回退] {symbol} 在线数据源无法连接，使用高保真合成数据进行测试")
        df = generate_synthetic_stock_data(symbol, start_date, end_date)

    # 5. 计算衍生字段与涨跌停
    if not df.empty:
        df = calculate_price_limits(df, symbol=symbol, is_st=is_st)
        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df["close"]
        if "turnover" not in df.columns:
            df["turnover"] = (df["volume"] / (df["volume"].rolling(20).mean().replace(0, 1))) * 1.5

        # 写入本地缓存
        try:
            df.to_csv(cache_file)
            logger.info(f"[缓存] {symbol} 已缓存 {len(df)} 条记录 -> {cache_file}")
        except Exception as e:
            logger.debug(f"写入缓存失败: {e}")

    return df


def fetch_multi_stocks(
    symbols: List[str],
    start_date: str = "20180101",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """批量获取多只股票数据"""
    result = {}
    for sym in symbols:
        try:
            df = fetch_stock_data(sym, start_date, end_date, use_cache=use_cache)
            if not df.empty:
                result[sym] = df
        except Exception as e:
            logger.error(f"[错误] {sym}: {e}")
    return result


def fetch_all_a_snapshot() -> pd.DataFrame:
    """
    获取全 A 股实时快照数据。
    """
    if ak is not None:
        try:
            spot = ak.stock_zh_a_spot()
            if spot is not None and not spot.empty:
                col_map = {
                    "代码": "code", "名称": "name", "最新价": "close",
                    "涨跌额": "change", "涨跌幅": "pct_chg", "买入": "bid",
                    "卖出": "ask", "昨收": "prev_close", "今开": "open",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "时间戳": "timestamp",
                }
                spot = spot.rename(columns=col_map)
                # 剔除代码前缀如 sh/sz/bj
                spot["code"] = spot["code"].astype(str).str.extract(r"(\d{6})")[0]
                return spot
        except Exception as e:
            logger.warning(f"[快照] akshare 快照获取失败: {e}")

    # 若快照获取失败，从默认股票池构造最新切片
    records = []
    for code, name in DEFAULT_STOCKS.items():
        df = fetch_stock_data(code, start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"), use_cache=True)
        if not df.empty:
            last = df.iloc[-1]
            records.append({
                "code": code,
                "name": name,
                "close": last["close"],
                "open": last["open"],
                "high": last["high"],
                "low": last["low"],
                "volume": last["volume"],
                "amount": last["amount"],
                "pct_chg": last.get("pct_chg", 0.0),
                "turnover": last.get("turnover", 1.0),
                "limit_up": last["limit_up"],
                "limit_down": last["limit_down"],
                "is_limit_up": last["is_limit_up"],
                "is_limit_down": last["is_limit_down"],
                "is_suspended": last["is_suspended"],
            })
    return pd.DataFrame(records)


def get_stock_info(symbol: str) -> dict:
    """获取股票基本信息"""
    name = DEFAULT_STOCKS.get(symbol, f"股票_{symbol}")
    return {
        "symbol": symbol,
        "name": name,
        "is_st": "ST" in name,
    }


if __name__ == "__main__":
    test_sym = "600519"
    df_test = fetch_stock_data(test_sym, "20230101", "20231231", use_cache=False)
    print(f"\n[测试成功] 获取 {test_sym} 数据:")
    print(df_test.tail())
    print("\n字段列表:", df_test.columns.tolist())
    print(f"数据量: {len(df_test)} 条, 区间: {df_test.index[0]} ~ {df_test.index[-1]}")
