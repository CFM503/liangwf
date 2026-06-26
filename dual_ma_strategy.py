#!/usr/bin/env python3
# ============================================================
#  A股双均线交叉量化策略 (dual_ma_strategy.py)
#  基于 Backtrader 回测框架
#
#  策略逻辑:
#    - MA5 上穿 MA20 (金叉) + 成交量放大 → 买入
#    - MA5 下穿 MA20 (死叉) 或 止损/止盈触发 → 卖出
#    - 跟踪止盈: 价格从最高点回落一定比例则平仓
#
#  风控规则:
#    - 单股仓位 ≤ 总资金 20%
#    - 总仓位 ≤ 60%
#    - 单笔止损 -8%
#    - 单笔止盈 +15%
#    - 跟踪止盈: 从盈利最高点回落 5% 触发
#
#  免责声明: 仅供学习研究，不构成投资建议。股市有风险，入市需谨慎。
# ============================================================

import os
import sys
import time
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import backtrader as bt
import backtrader.analyzers as btanalyzers
import akshare as ak
from tabulate import tabulate

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 第一部分: 策略参数配置
# ============================================================

CONFIG = {
    # --- 股票池 ---
    "stocks": {
        "600519": "贵州茅台",
        "000858": "五粮液",
        "601318": "中国平安",
        "600036": "招商银行",
        "000001": "平安银行",
    },
    "start_date": "20180101",
    "end_date": "20250630",

    # --- 均线参数 ---
    "ma_fast": 5,          # 短期均线周期
    "ma_slow": 20,         # 长期均线周期

    # --- 成交量过滤 ---
    "vol_period": 20,      # 均量计算周期
    "vol_multiplier": 1.5, # 量比阈值

    # --- 仓位控制 ---
    "single_stock_pct": 0.20,   # 单股最大仓位 20%
    "total_position_pct": 0.60, # 总仓位上限 60%

    # --- 止损止盈 ---
    "stop_loss_pct": 0.08,       # 止损 -8%
    "take_profit_pct": 0.15,     # 止盈 +15%
    "trailing_stop_pct": 0.05,   # 跟踪止盈: 从高点回落 5%
    "trailing_start_pct": 0.08,  # 盈利超过 8% 才启动跟踪止盈

    # --- 回测参数 ---
    "initial_cash": 1_000_000,   # 初始资金 100万
    "commission": 0.0003,        # 佣金万三
    "stamp_tax": 0.001,          # 印花税千一(卖出)
    "slippage_pct": 0.001,       # 滑点 0.1%
}


# ============================================================
# 第二部分: 数据获取 (Akshare)
# ============================================================


import requests


def _generate_synthetic_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    生成模拟A股日K线数据（用于网络不通时演示回测）

    基于真实A股统计特征:
      - 年化波动率 25%-40%
      - 日均换手率特征
      - 含趋势、震荡、急涨急跌多种行情
    """
    np.random.seed(hash(stock_code) % 2**31)

    # 不同股票不同起始价格和波动特征
    stock_profiles = {
        "600519": {"base_price": 680, "annual_vol": 0.28, "drift": 0.08, "name": "贵州茅台"},
        "000858": {"base_price": 150, "annual_vol": 0.35, "drift": 0.05, "name": "五粮液"},
        "601318": {"base_price": 55,  "annual_vol": 0.30, "drift": 0.03, "name": "中国平安"},
        "600036": {"base_price": 35,  "annual_vol": 0.28, "drift": 0.06, "name": "招商银行"},
        "000001": {"base_price": 12,  "annual_vol": 0.32, "drift": 0.02, "name": "平安银行"},
    }
    profile = stock_profiles.get(stock_code, {"base_price": 50, "annual_vol": 0.30, "drift": 0.04})

    # 生成交易日序列（排除周末）
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)

    base_price = profile["base_price"]
    daily_vol = profile["annual_vol"] / np.sqrt(252)
    daily_drift = profile["drift"] / 252

    # 用GBM + 均值回复 + 跳跃生成价格路径
    prices = [base_price]
    for i in range(1, n):
        # 基础随机游走
        ret = daily_drift + daily_vol * np.random.randn()

        # 加入趋势切换（模拟牛熊转换）
        cycle = np.sin(2 * np.pi * i / 500) * 0.0005
        ret += cycle

        # 加入跳跃（模拟涨停跌停、突发消息）
        if np.random.random() < 0.02:
            ret += np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.07)

        # 均值回复（防止价格飞天）
        if len(prices) > 60:
            ma60 = np.mean(prices[-60:])
            ret -= 0.001 * (prices[-1] / ma60 - 1)

        new_price = prices[-1] * (1 + ret)
        new_price = max(new_price, base_price * 0.3)  # 下限
        prices.append(new_price)

    prices = np.array(prices)

    # 生成OHLCV
    opens = prices * (1 + np.random.randn(n) * 0.005)
    highs = np.maximum(prices, opens) * (1 + np.abs(np.random.randn(n)) * 0.008)
    lows = np.minimum(prices, opens) * (1 - np.abs(np.random.randn(n)) * 0.008)
    closes = prices

    # 成交量：对数正态 + 与涨跌相关
    base_vol = np.random.lognormal(mean=16, sigma=0.5)
    volume_noise = np.random.lognormal(mean=0, sigma=0.4, size=n)
    # 涨跌时放量
    returns = np.diff(closes, prepend=closes[0]) / closes
    vol_multiplier = 1 + np.abs(returns) * 10
    volumes = base_vol * volume_noise * vol_multiplier

    # 限制涨跌幅 ±10%（A股主板）
    for i in range(1, n):
        change = (closes[i] - closes[i-1]) / closes[i-1]
        if abs(change) > 0.10:
            closes[i] = closes[i-1] * (1 + np.sign(change) * 0.10)

    df = pd.DataFrame({
        "date": dates[:n],
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": np.round(volumes).astype(int),
    })

    logger.info(f"[模拟数据] {stock_code} 生成 {len(df)} 条 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
    return df


def _fetch_from_eastmoney(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    直接调用东方财富API获取日K线（akshare底层也是这个接口）

    绕过 akshare 的连接问题，直接用 requests 请求
    """
    # 东方财富 secid: 0=深圳, 1=上海
    market = "1" if stock_code.startswith("6") else "0"
    secid = f"{market}.{stock_code}"

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",         # 日K
        "fqt": "1",           # 前复权
        "beg": start_date,
        "end": end_date,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com",
    }

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    klines = data.get("data", {}).get("klines", [])
    if not klines:
        return pd.DataFrame()

    records = []
    for line in klines:
        parts = line.split(",")
        records.append({
            "date": parts[0],
            "open": float(parts[1]),
            "close": float(parts[2]),
            "high": float(parts[3]),
            "low": float(parts[4]),
            "volume": float(parts[5]),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_stock_data(stock_code: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    """
    获取A股日K线数据（前复权）

    优先用 akshare，失败后直接调东方财富API，再失败用本地CSV
    """
    # 方式1: akshare
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
            logger.info(f"[akshare] {stock_code} 成功: {len(df)} 条")
            return df
        except Exception as e:
            logger.warning(f"[akshare] {stock_code} 第{attempt+1}次失败: {e}")
            time.sleep(1)

    # 方式2: 直接调东方财富API
    try:
        logger.info(f"[东方财富API] 正在获取 {stock_code}...")
        df = _fetch_from_eastmoney(stock_code, start_date, end_date)
        if not df.empty:
            logger.info(f"[东方财富API] {stock_code} 成功: {len(df)} 条")
            return df
    except Exception as e:
        logger.warning(f"[东方财富API] {stock_code} 失败: {e}")

    # 方式3: 本地CSV回退
    csv_path = f"data/{stock_code}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["date"])
        logger.info(f"[CSV] {stock_code} 从本地加载: {len(df)} 条")
        return df

    # 方式4: 内置模拟数据（网络不通时用于演示回测）
    logger.warning(f"网络不通, 为 {stock_code} 生成模拟数据用于策略验证")
    return _generate_synthetic_data(stock_code, start_date, end_date)


def load_all_data(config: dict) -> dict:
    """加载所有股票数据, 返回 {code: DataFrame}"""
    data = {}
    for code in config["stocks"]:
        df = fetch_stock_data(code, config["start_date"], config["end_date"])
        if not df.empty and len(df) > 100:
            data[code] = df
        time.sleep(0.8)  # 请求间隔加大
    return data


# ============================================================
# 第三部分: Backtrader 数据适配器
# ============================================================


class PandasData(bt.feeds.PandasData):
    """
    将 DataFrame 适配为 Backtrader 数据源

    注意: datetime=None 表示使用 DataFrame 的 index 作为日期
    """
    params = (
        ("datetime", None),    # None = 使用 index
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


# ============================================================
# 第四部分: 自定义佣金方案 (含印花税)
# ============================================================


class AShareCommission(bt.CommInfoBase):
    """
    A股佣金方案:
      - 买入: 佣金 (万三, 最低5元)
      - 卖出: 佣金 + 印花税 (千一)
    """
    params = (
        ("commission", 0.0003),
        ("stamp_tax", 0.001),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        计算佣金:
          买入 = 成交额 * 佣金率 (最低5元)
          卖出 = 成交额 * (佣金率 + 印花税率) (最低5元)
        """
        turnover = abs(size) * price
        comm = turnover * self.p.commission
        comm = max(comm, 5.0)

        # 卖出时加收印花税
        if size < 0:
            comm += turnover * self.p.stamp_tax

        return comm


# ============================================================
# 第五部分: 核心策略 (双均线交叉 + 成交量过滤 + 止损止盈)
# ============================================================


class DualMAStrategy(bt.Strategy):
    """
    双均线交叉策略

    买入条件 (三重确认):
      1. MA5 上穿 MA20 (金叉)
      2. 当日成交量 > 20日均量 * 1.5 (放量确认)
      3. 仓位检查通过 (单股 ≤ 20%, 总仓位 ≤ 60%)

    卖出条件 (任一触发):
      1. MA5 下穿 MA20 (死叉)
      2. 亏损达到 -8% (硬止损)
      3. 盈利达到 +15% (止盈)
      4. 盈利 > 8% 后, 从最高点回落 5% (跟踪止盈)
    """

    params = dict(
        ma_fast=5,
        ma_slow=20,
        vol_period=20,
        vol_multiplier=1.5,
        single_stock_pct=0.20,
        total_position_pct=0.60,
        stop_loss_pct=0.08,
        take_profit_pct=0.15,
        trailing_stop_pct=0.05,
        trailing_start_pct=0.08,
        printlog=True,
    )

    def __init__(self):
        self.orders = {}           # {data: order} 待执行订单
        self.buy_prices = {}       # {data: 买入价格}
        self.buy_dates = {}        # {data: 买入日期}
        self.highest_since_buy = {}  # {data: 买入后最高价}
        self.trade_log = []        # 成交记录
        self.daily_values = []     # 每日净值

        # 为每只股票计算指标
        self.indicators = {}
        for d in self.datas:
            self.indicators[d] = {
                "ma_fast": bt.indicators.SMA(d.close, period=self.params.ma_fast),
                "ma_slow": bt.indicators.SMA(d.close, period=self.params.ma_slow),
                "crossover": bt.indicators.CrossOver(
                    bt.indicators.SMA(d.close, period=self.params.ma_fast),
                    bt.indicators.SMA(d.close, period=self.params.ma_slow),
                ),
                "vol_avg": bt.indicators.SMA(d.volume, period=self.params.vol_period),
            }

    def log(self, txt, dt=None):
        """日志输出"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            logging.debug(f"[{dt}] {txt}")

    def notify_order(self, order):
        """订单状态回调"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        data = order.data
        dt = self.datas[0].datetime.date(0)

        if order.status == order.Completed:
            if order.isbuy():
                self.buy_prices[data] = order.executed.price
                self.buy_dates[data] = dt
                self.highest_since_buy[data] = order.executed.price
                self.log(
                    f"买入 {data._name} | 价格: {order.executed.price:.2f} | "
                    f"数量: {order.executed.size:.0f} | 费用: {order.executed.comm:.2f}"
                )
            elif order.issell():
                buy_price = self.buy_prices.get(data, 0)
                pnl = (order.executed.price - buy_price) * order.executed.size
                pnl_pct = (order.executed.price / buy_price - 1) * 100 if buy_price else 0

                self.trade_log.append({
                    "股票": data._name,
                    "买入日期": self.buy_dates.get(data, dt),
                    "买入价": buy_price,
                    "卖出日期": dt,
                    "卖出价": order.executed.price,
                    "数量": abs(order.executed.size),
                    "盈亏": pnl,
                    "盈亏%": pnl_pct,
                    "佣金": order.executed.comm,
                })

                self.log(
                    f"卖出 {data._name} | 价格: {order.executed.price:.2f} | "
                    f"盈亏: {pnl:+.0f} ({pnl_pct:+.1f}%)"
                )
                # 清理记录
                self.buy_prices.pop(data, None)
                self.buy_dates.pop(data, None)
                self.highest_since_buy.pop(data, None)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单被拒/取消: {data._name} {order.status}")

        self.orders.pop(data, None)

    def notify_trade(self, trade):
        """交易完成回调"""
        pass

    def _get_position_value(self):
        """计算当前总持仓市值"""
        total = 0
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                total += pos.size * d.close[0]
        return total

    def _get_total_value(self):
        """计算总资产"""
        return self.broker.getvalue()

    def _can_buy(self, data):
        """
        仓位检查:
          - 单股仓位 ≤ 20%
          - 总仓位 ≤ 60%
        """
        total_value = self._get_total_value()
        position_value = self._get_position_value()

        # 总仓位检查
        if position_value >= total_value * self.params.total_position_pct:
            return False

        # 单股仓位检查
        current_pos = self.getposition(data)
        current_value = current_pos.size * data.close[0] if current_pos.size > 0 else 0
        max_single = total_value * self.params.single_stock_pct
        if current_value >= max_single * 0.95:  # 接近上限也不加仓
            return False

        return True

    def _calculate_buy_size(self, data):
        """
        计算买入数量 (A股100股整数倍)

        取可用仓位空间与单股上限的较小值
        """
        total_value = self._get_total_value()
        position_value = self._get_position_value()

        # 剩余可用仓位空间
        remaining_total = total_value * self.params.total_position_pct - position_value
        # 单股剩余空间
        current_pos = self.getposition(data)
        current_value = current_pos.size * data.close[0] if current_pos.size > 0 else 0
        remaining_single = total_value * self.params.single_stock_pct - current_value

        alloc = min(remaining_total, remaining_single)
        if alloc <= 0:
            return 0

        price = data.close[0]
        shares = int(alloc / price / 100) * 100
        return max(shares, 0)

    def _check_stop_conditions(self, data):
        """
        检查止损止盈条件

        返回: (should_sell: bool, reason: str)
        """
        if data not in self.buy_prices:
            return False, ""

        buy_price = self.buy_prices[data]
        current_price = data.close[0]
        pnl_pct = (current_price - buy_price) / buy_price

        # 1. 硬止损: -8%
        if pnl_pct <= -self.params.stop_loss_pct:
            return True, f"止损({pnl_pct:.1%})"

        # 2. 止盈: +15%
        if pnl_pct >= self.params.take_profit_pct:
            return True, f"止盈({pnl_pct:.1%})"

        # 3. 跟踪止盈
        # 更新买入后最高价
        if data in self.highest_since_buy:
            self.highest_since_buy[data] = max(self.highest_since_buy[data], current_price)
        else:
            self.highest_since_buy[data] = current_price

        # 只有盈利超过 trailing_start_pct (8%) 才启动跟踪止盈
        if pnl_pct >= self.params.trailing_start_pct:
            highest = self.highest_since_buy[data]
            drawback = (highest - current_price) / highest
            if drawback >= self.params.trailing_stop_pct:
                return True, f"跟踪止盈(高点回落{drawback:.1%})"

        return False, ""

    def next(self):
        """主策略逻辑 - 每个交易日调用一次"""
        dt = self.datas[0].datetime.date(0)

        for d in self.datas:
            ind = self.indicators[d]

            # --- 跳过数据不足的日期 ---
            if len(d) < self.params.ma_slow + 1:
                continue

            # --- 检查止损止盈 (优先级最高) ---
            has_position = self.getposition(d).size > 0
            if has_position:
                should_sell, reason = self._check_stop_conditions(d)
                if should_sell:
                    self.log(f"[风控] {d._name} 触发 {reason}")
                    self.close(d)
                    continue

            # --- 检查是否有待处理订单 ---
            if d in self.orders:
                continue

            # --- 均线交叉信号 ---
            crossover = ind["crossover"][0]
            vol_today = d.volume[0]
            vol_avg = ind["vol_avg"][0]
            vol_ok = vol_today > vol_avg * self.params.vol_multiplier

            # --- 买入信号: 金叉 + 放量 ---
            if crossover > 0 and vol_ok:
                if not has_position and self._can_buy(d):
                    size = self._calculate_buy_size(d)
                    if size > 0:
                        self.log(
                            f"[买入] {d._name} 金叉+放量 | "
                            f"MA{self.params.ma_fast}={ind['ma_fast'][0]:.2f} > "
                            f"MA{self.params.ma_slow}={ind['ma_slow'][0]:.2f} | "
                            f"量比={vol_today / vol_avg:.1f}x | 买入 {size} 股"
                        )
                        self.orders[d] = self.buy(d, size=size)

            # --- 卖出信号: 死叉 ---
            elif crossover < 0:
                if has_position:
                    self.log(f"[卖出] {d._name} 死叉信号")
                    self.orders[d] = self.sell(d, size=self.getposition(d).size)

        # --- 记录每日净值 ---
        self.daily_values.append({
            "date": dt,
            "value": self._get_total_value(),
            "cash": self.broker.getcash(),
            "position": self._get_position_value(),
        })

    def stop(self):
        """回测结束时调用"""
        final = self.broker.getvalue()
        logger.info(f"回测结束 | 最终资产: {final:,.0f}")


# ============================================================
# 第六部分: 回测引擎
# ============================================================


def run_backtest(config: dict, printlog: bool = False) -> dict:
    """
    执行回测

    参数:
        config: 策略配置
        printlog: 是否打印逐日交易日志
    返回:
        绩效指标字典
    """
    cerebro = bt.Cerebro()

    # 1. 加载数据
    logger.info("正在获取股票数据...")
    all_data = load_all_data(config)
    if not all_data:
        logger.error("无有效数据，退出")
        sys.exit(1)

    logger.info(f"共加载 {len(all_data)} 只股票数据")

    # 2. 添加数据到 Cerebro
    for code, df in all_data.items():
        # Backtrader 需要 date 为索引
        df_bt = df.set_index("date")
        data_feed = PandasData(
            dataname=df_bt,
            name=code,
        )
        cerebro.adddata(data_feed)

    # 3. 设置策略参数
    cerebro.addstrategy(
        DualMAStrategy,
        ma_fast=config["ma_fast"],
        ma_slow=config["ma_slow"],
        vol_period=config["vol_period"],
        vol_multiplier=config["vol_multiplier"],
        single_stock_pct=config["single_stock_pct"],
        total_position_pct=config["total_position_pct"],
        stop_loss_pct=config["stop_loss_pct"],
        take_profit_pct=config["take_profit_pct"],
        trailing_stop_pct=config["trailing_stop_pct"],
        trailing_start_pct=config["trailing_start_pct"],
        printlog=printlog,
    )

    # 4. 设置资金和佣金
    cerebro.broker.setcash(config["initial_cash"])
    cerebro.broker.addcommissioninfo(AShareCommission(
        commission=config["commission"],
        stamp_tax=config["stamp_tax"],
    ))
    cerebro.broker.set_slippage_perc(config["slippage_pct"])

    # 5. 添加分析器
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name="annual_return")
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.02)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    cerebro.addanalyzer(btanalyzers.TimeReturn, _name="time_return")

    # 6. 执行回测
    logger.info(f"初始资金: {config['initial_cash']:,.0f}")
    logger.info("开始回测...")
    start_time = time.time()
    results = cerebro.run()
    elapsed = time.time() - start_time
    logger.info(f"回测完成, 耗时: {elapsed:.1f}秒")

    strat = results[0]

    # 7. 提取绩效指标
    metrics = extract_metrics(strat, config)

    # 8. 绘制图表
    plot_results(strat, config, metrics)

    return metrics


# ============================================================
# 第七部分: 绩效指标计算
# ============================================================


def extract_metrics(strat, config: dict) -> dict:
    """从策略结果中提取全部绩效指标"""
    initial = config["initial_cash"]

    # 净值序列
    daily = pd.DataFrame(strat.daily_values)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")

    final_value = daily["value"].iloc[-1]
    total_return = (final_value - initial) / initial
    trading_days = len(daily)
    years = trading_days / 252

    # 年化收益
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 夏普比率
    daily_returns = daily["value"].pct_change().dropna()
    sharpe = 0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() - 0.02 / 252) / daily_returns.std() * np.sqrt(252)

    # 最大回撤
    cummax = daily["value"].cummax()
    drawdown = (cummax - daily["value"]) / cummax
    max_drawdown = drawdown.max()
    max_dd_date = drawdown.idxmax()

    # 交易统计
    trades = pd.DataFrame(strat.trade_log)
    total_trades = len(trades)
    win_trades = 0
    lose_trades = 0
    avg_win = 0
    avg_loss = 0
    win_rate = 0
    profit_factor = 0

    if total_trades > 0:
        win_trades = len(trades[trades["盈亏"] > 0])
        lose_trades = len(trades[trades["盈亏"] <= 0])
        win_rate = win_trades / total_trades

        wins = trades[trades["盈亏"] > 0]["盈亏"]
        losses = trades[trades["盈亏"] <= 0]["盈亏"]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

    # 按股票统计
    stock_stats = {}
    if total_trades > 0:
        for stock, group in trades.groupby("股票"):
            s_wins = len(group[group["盈亏"] > 0])
            s_total = len(group)
            stock_stats[stock] = {
                "交易次数": s_total,
                "胜率": f"{s_wins / s_total:.0%}" if s_total > 0 else "N/A",
                "总盈亏": f"{group['盈亏'].sum():+,.0f}",
            }

    metrics = {
        "初始资金": f"{initial:,.0f}",
        "最终权益": f"{final_value:,.0f}",
        "总收益率": f"{total_return:.2%}",
        "年化收益率": f"{annual_return:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_drawdown:.2%}",
        "最大回撤日期": str(max_dd_date.date()),
        "交易天数": trading_days,
        "回测年数": f"{years:.1f}",
        "总交易笔数": total_trades,
        "盈利笔数": win_trades,
        "亏损笔数": lose_trades,
        "胜率": f"{win_rate:.2%}",
        "平均盈利": f"{avg_win:+,.0f}",
        "平均亏损": f"{avg_loss:+,.0f}",
        "盈亏比": f"{profit_factor:.2f}",
        "_daily": daily,
        "_trades": trades,
        "_stock_stats": stock_stats,
    }
    return metrics


# ============================================================
# 第八部分: 报告输出与可视化
# ============================================================


def print_report(metrics: dict):
    """打印完整回测报告"""
    print()
    print("=" * 65)
    print("  A股双均线交叉策略 — 回测报告")
    print("  [!] 风险警告: 回测结果不代表未来收益，仅供学习参考")
    print("=" * 65)

    # 核心指标
    keys = [
        "初始资金", "最终权益", "总收益率", "年化收益率", "夏普比率",
        "最大回撤", "最大回撤日期", "交易天数", "回测年数",
        "总交易笔数", "盈利笔数", "亏损笔数", "胜率",
        "平均盈利", "平均亏损", "盈亏比",
    ]
    table = [[k, metrics[k]] for k in keys]
    print(tabulate(table, headers=["指标", "数值"], tablefmt="grid"))

    # 分股票统计
    if metrics["_stock_stats"]:
        print("\n--- 分股票统计 ---")
        stock_table = [[code, *stats.values()] for code, stats in metrics["_stock_stats"].items()]
        print(tabulate(stock_table, headers=["股票", "交易次数", "胜率", "总盈亏"], tablefmt="grid"))

    # 最近10笔交易
    trades = metrics["_trades"]
    if not trades.empty:
        print("\n--- 最近 10 笔交易 ---")
        recent = trades.tail(10).copy()
        for col in ["买入价", "卖出价"]:
            recent[col] = recent[col].apply(lambda x: f"{x:.2f}")
        recent["盈亏"] = recent["盈亏"].apply(lambda x: f"{x:+,.0f}")
        recent["盈亏%"] = recent["盈亏%"].apply(lambda x: f"{x:+.1f}%")
        print(tabulate(recent.values.tolist(), headers=recent.columns.tolist(), tablefmt="grid"))

    print()


def plot_results(strat, config: dict, metrics: dict):
    """绘制回测结果图表并保存为PNG"""
    daily = metrics["_daily"]
    trades = metrics["_trades"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    fig.suptitle(
        f"Dual MA Crossover Strategy Backtest\n"
        f"MA{config['ma_fast']}/{config['ma_slow']} | Vol Filter {config['vol_multiplier']}x | "
        f"SL {config['stop_loss_pct']:.0%} TP {config['take_profit_pct']:.0%} | "
        f"Return: {metrics['总收益率']} Sharpe: {metrics['夏普比率']}",
        fontsize=13, fontweight="bold",
    )

    # 图1: 净值曲线
    ax1 = axes[0]
    ax1.plot(daily.index, daily["value"], color="#1976D2", linewidth=1.5, label="Portfolio Value")
    ax1.axhline(y=config["initial_cash"], color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax1.fill_between(daily.index, daily["value"], config["initial_cash"],
                     where=daily["value"] >= config["initial_cash"], alpha=0.1, color="#4CAF50")
    ax1.fill_between(daily.index, daily["value"], config["initial_cash"],
                     where=daily["value"] < config["initial_cash"], alpha=0.1, color="#F44336")

    # 标注买卖点
    if not trades.empty:
        for _, t in trades.iterrows():
            ax1.axvline(x=pd.Timestamp(t["买入日期"]), color="#4CAF50", alpha=0.15, linewidth=0.5)
            ax1.axvline(x=pd.Timestamp(t["卖出日期"]), color="#F44336", alpha=0.15, linewidth=0.5)

    ax1.set_ylabel("Portfolio Value (CNY)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Equity Curve")

    # 图2: 回撤
    ax2 = axes[1]
    cummax = daily["value"].cummax()
    drawdown = (cummax - daily["value"]) / cummax * 100
    ax2.fill_between(daily.index, drawdown, 0, color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)

    # 图3: 仓位
    ax3 = axes[2]
    pos_pct = daily["position"] / daily["value"] * 100
    ax3.fill_between(daily.index, pos_pct, 0, color="#2196F3", alpha=0.4)
    ax3.axhline(y=config["total_position_pct"] * 100, color="red", linestyle="--", alpha=0.5,
                label=f"Max Position {config['total_position_pct']:.0%}")
    ax3.set_ylabel("Position (%)")
    ax3.set_title("Position Ratio")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    # 图4: 单笔盈亏分布
    ax4 = axes[3]
    if not trades.empty:
        colors = ["#4CAF50" if x > 0 else "#F44336" for x in trades["盈亏%"]]
        ax4.bar(range(len(trades)), trades["盈亏%"], color=colors, alpha=0.7)
        ax4.axhline(y=0, color="black", linewidth=0.5)
        ax4.axhline(y=config["take_profit_pct"] * 100, color="green", linestyle="--", alpha=0.3, label=f"TP +{config['take_profit_pct']:.0%}")
        ax4.axhline(y=-config["stop_loss_pct"] * 100, color="red", linestyle="--", alpha=0.3, label=f"SL -{config['stop_loss_pct']:.0%}")
        ax4.set_xlabel("Trade #")
        ax4.set_ylabel("P&L (%)")
        ax4.set_title("Per-Trade P&L")
        ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "dual_ma_backtest.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"图表已保存: {save_path}")


# 需要 import matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 第九部分: 参数优化建议
# ============================================================


def print_optimization_guide():
    """输出参数优化建议和风险分析"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    参数优化建议 & 风险分析                        ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│ 一、参数优化方向                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. 均线周期优化                                              │
│    - 快线: 3/5/10 日  ← 更短=更灵敏, 更多交易, 更多噪音       │
│    - 慢线: 10/20/30/60 日 ← 更长=更稳定, 更少交易, 滞后更大     │
│    - 建议: 在 [3,5,10] x [15,20,30,60] 网格搜索               │
│                                                             │
│ 2. 成交量阈值优化                                            │
│    - 1.2x ~ 2.0x 之间测试                                   │
│    - 过高(>2.5x): 信号太少，错过行情                          │
│    - 过低(<1.2x): 过滤效果差，假信号多                         │
│                                                             │
│ 3. 止损止盈比例                                              │
│    - 止损: 5%-10% (过小容易被洗出, 过大亏损深)                  │
│    - 止盈: 10%-25% (配合跟踪止盈使用)                          │
│    - 跟踪止盈启动点: 5%-12%                                   │
│    - 跟踪回撤: 3%-8%                                         │
│                                                             │
│ 4. 仓位比例                                                 │
│    - 激进: 单股30%, 总仓80%                                   │
│    - 稳健: 单股15%, 总仓50% ← 推荐                            │
│    - 保守: 单股10%, 总仓30%                                   │
│                                                             │
│ 5. Backtrader内置优化器                                      │
│    cerebro.optstrategy(                                     │
│        DualMAStrategy,                                      │
│        ma_fast=[3, 5, 10],                                  │
│        ma_slow=[15, 20, 30, 60],                            │
│        vol_multiplier=[1.2, 1.5, 2.0],                      │
│    )                                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 二、潜在风险分析                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. 【滞后性风险】                                             │
│    均线是滞后指标，金叉时往往已涨一段，死叉时已跌一段。          │
│    在震荡市中反复假突破 → 频繁止损 → 资金磨损。                 │
│                                                             │
│ 2. 【震荡市风险】                                             │
│    双均线策略本质是趋势跟踪，横盘震荡期间表现最差。              │
│    A股2018-2024有多段长期横盘，期间可能持续亏损。               │
│                                                             │
│ 3. 【过拟合风险】                                             │
│    参数优化到历史数据最优 ≠ 未来最优。                          │
│    MA5/20 只是默认值，不应作为唯一真理。                       │
│                                                             │
│ 4. 【流动性风险】                                             │
│    小盘股在极端行情下可能无法以目标价成交。                      │
│    滑点模型 0.1% 较乐观，实际可能更大。                         │
│                                                             │
│ 5. 【幸存者偏差】                                             │
│    回测标的多为大盘蓝筹(茅台/平安等)，                          │
│    这些股票本身长期上涨，不能代表全市场表现。                    │
│                                                             │
│ 6. 【制度风险】                                               │
│    T+1限制、涨跌停板、停牌等A股特有制度会影响策略执行。          │
│    本回测未完全模拟涨跌停限制。                                │
│                                                             │
│ 7. 【资金容量风险】                                           │
│    100万资金的策略表现 ≠ 1亿资金的表现。                       │
│    大资金的冲击成本会显著降低收益。                             │
│                                                             │
│ 【建议】                                                     │
│  - 先用模拟盘验证至少3个月再考虑实盘                           │
│  - 定期检视策略有效性，市场风格会切换                          │
│  - 分散策略: 不要只用双均线，配合其他策略使用                    │
│  - 控制预期: 年化15-25%已是优秀水平，承诺50%+的都是骗局         │
└─────────────────────────────────────────────────────────────┘
""")


# ============================================================
# 第十部分: 主入口
# ============================================================


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="A股双均线交叉策略回测")
    parser.add_argument("--fast", type=int, default=None, help="快线周期 (覆盖配置)")
    parser.add_argument("--slow", type=int, default=None, help="慢线周期 (覆盖配置)")
    parser.add_argument("--vol", type=float, default=None, help="量比阈值 (覆盖配置)")
    parser.add_argument("--verbose", action="store_true", help="打印逐日交易日志")
    parser.add_argument("--guide", action="store_true", help="只打印优化建议")
    args = parser.parse_args()

    if args.guide:
        print_optimization_guide()
        return

    # 覆盖配置
    cfg = CONFIG.copy()
    if args.fast:
        cfg["ma_fast"] = args.fast
    if args.slow:
        cfg["ma_slow"] = args.slow
    if args.vol:
        cfg["vol_multiplier"] = args.vol

    # 运行回测
    metrics = run_backtest(cfg, printlog=args.verbose)

    # 打印报告
    print_report(metrics)

    # 打印优化建议
    print_optimization_guide()


if __name__ == "__main__":
    main()
