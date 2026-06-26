#!/usr/bin/env python3
# ============================================================
#  A股程序化量化交易系统 (quant_system.py)
#  灵感来源：梁文锋 (幻方量化 / DeepSeek 创始人) 早期策略思路
#
#  核心理念：
#    1. 数据驱动，消除人为情绪干扰
#    2. 双均线交叉 + 成交量过滤 + ML预测 三重信号确认
#    3. 严格风控：止损止盈、仓位管理、日亏损限额
#
#  免责声明：
#    本程序仅供学习研究，不构成任何投资建议。
#    股市有风险，入市需谨慎。回测结果不代表未来收益。
#
#  使用方式：
#    pip install akshare pandas numpy scikit-learn matplotlib pyyaml tabulate
#    python quant_system.py                    # 运行回测
#    python quant_system.py --mode live        # 实盘模式（需配置券商API）
# ============================================================

import os
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无GUI后端，兼容服务器环境
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ============================================================
# 第一部分：配置与数据结构
# ============================================================


class Signal(Enum):
    """交易信号枚举"""

    BUY = 1
    SELL = -1
    HOLD = 0


class OrderSide(Enum):
    """订单方向"""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """订单数据结构"""

    stock_code: str
    side: OrderSide
    price: float
    quantity: int
    timestamp: str
    reason: str = ""
    status: str = "pending"  # pending / filled / rejected


@dataclass
class Position:
    """持仓数据结构"""

    stock_code: str
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    def update_price(self, price: float):
        """更新当前价格并计算浮动盈亏"""
        self.current_price = price
        if self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_cost) * self.quantity


@dataclass
class TradeRecord:
    """成交记录"""

    stock_code: str
    side: str
    price: float
    quantity: int
    amount: float
    commission: float
    tax: float
    timestamp: str
    reason: str


def load_config(config_path: str = "strategy.yaml") -> dict:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        logging.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logging.info(f"已加载配置文件: {config_path}")
    return cfg


def get_default_config() -> dict:
    """默认配置（当无YAML文件时使用）"""
    return {
        "data": {
            "source": "akshare",
            "stock_pool": ["000001", "600519"],
            "start_date": "20230101",
            "end_date": "20251231",
            "frequency": "daily",
        },
        "strategy": {
            "name": "dual_ma_volume_ml",
            "ma": {"fast_period": 5, "slow_period": 20},
            "volume": {"enabled": True, "avg_period": 20, "multiplier": 1.5},
            "ml": {"enabled": True, "model": "random_forest", "lookback_days": 60, "train_ratio": 0.7},
        },
        "risk": {
            "position": {"max_position_pct": 0.3, "total_position_pct": 0.8, "min_trade_amount": 100},
            "stop_loss": {"enabled": True, "pct": 0.05},
            "take_profit": {"enabled": True, "pct": 0.10},
            "daily_loss": {"enabled": True, "max_daily_loss_pct": 0.03},
            "max_drawdown": {"enabled": True, "pct": 0.15},
        },
        "backtest": {
            "initial_capital": 1000000,
            "commission_rate": 0.0003,
            "stamp_tax_rate": 0.001,
            "slippage_pct": 0.001,
            "benchmark": "000300",
        },
        "logging": {"level": "INFO", "save_report": True, "report_dir": "./reports", "plot_results": True},
    }


# ============================================================
# 第二部分：数据获取模块
# ============================================================


class DataFetcher:
    """
    数据获取模块
    支持 akshare（免费，无需token）和 tushare（需pro token）
    """

    def __init__(self, config: dict):
        self.source = config["data"]["source"]
        self.tushare_token = config["data"].get("tushare_token", "")
        self._ts_api = None

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日K线数据

        返回的DataFrame包含列:
            date, open, high, low, close, volume, amount
        """
        if self.source == "akshare":
            return self._fetch_akshare(stock_code, start_date, end_date)
        elif self.source == "tushare":
            return self._fetch_tushare(stock_code, start_date, end_date)
        else:
            raise ValueError(f"不支持的数据源: {self.source}")

    def _fetch_akshare(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """通过akshare获取A股日K线"""
        import akshare as ak

        try:
            # akshare 接口：stock_zh_a_hist
            # stock_code 格式: 000001 (纯数字)
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
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
            df = df.sort_values("date").reset_index(drop=True)
            logging.info(f"[akshare] 获取 {stock_code} 数据成功, 共 {len(df)} 条")
            return df

        except Exception as e:
            logging.error(f"[akshare] 获取 {stock_code} 数据失败: {e}")
            return pd.DataFrame()

    def _fetch_tushare(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """通过tushare获取A股日K线"""
        import tushare as ts

        if self._ts_api is None:
            ts.set_token(self.tushare_token)
            self._ts_api = ts.pro_api()

        try:
            df = self._ts_api.daily(ts_code=f"{stock_code}.SH" if stock_code.startswith("6") else f"{stock_code}.SZ",
                                    start_date=start_date, end_date=end_date)
            df = df.rename(columns={"trade_date": "date", "vol": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            logging.info(f"[tushare] 获取 {stock_code} 数据成功, 共 {len(df)} 条")
            return df[["date", "open", "high", "low", "close", "volume", "amount"]]
        except Exception as e:
            logging.error(f"[tushare] 获取 {stock_code} 数据失败: {e}")
            return pd.DataFrame()

    def get_realtime_quote(self, stock_code: str) -> dict:
        """获取实时行情（用于实盘）"""
        import akshare as ak

        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == stock_code]
            if row.empty:
                return {}
            row = row.iloc[0]
            return {
                "code": stock_code,
                "name": row["名称"],
                "price": float(row["最新价"]),
                "change_pct": float(row["涨跌幅"]),
                "volume": float(row["成交量"]),
                "amount": float(row["成交额"]),
            }
        except Exception as e:
            logging.error(f"获取实时行情失败: {e}")
            return {}


# ============================================================
# 第三部分：技术指标计算模块
# ============================================================


class Indicators:
    """技术指标计算工具类"""

    @staticmethod
    def moving_average(series: pd.Series, period: int) -> pd.Series:
        """简单移动平均线 (SMA)"""
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def exponential_ma(series: pd.Series, period: int) -> pd.Series:
        """指数移动平均线 (EMA)"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指标 (RSI)"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD指标，返回 (DIF, DEA, HIST)"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = 2 * (dif - dea)
        return dif, dea, hist

    @staticmethod
    def volatility(series: pd.Series, period: int = 20) -> pd.Series:
        """历史波动率（收益率标准差）"""
        returns = series.pct_change()
        return returns.rolling(window=period, min_periods=1).std() * np.sqrt(252)

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """量比 = 当日成交量 / 过去N日平均成交量"""
        avg_vol = volume.rolling(window=period, min_periods=1).mean()
        return volume / (avg_vol + 1e-10)


# ============================================================
# 第四部分：特征工程与机器学习模块
# ============================================================


class MLPredictor:
    """
    机器学习预测模块

    使用技术指标作为特征，预测次日涨跌方向
    模型：随机森林 / 梯度提升
    """

    def __init__(self, config: dict):
        self.ml_cfg = config["strategy"]["ml"]
        self.model = None
        self.feature_cols = self.ml_cfg.get("feature_set", [])
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算ML所需的全部特征列

        参数:
            df: 包含 date, open, high, low, close, volume 的DataFrame
        返回:
            添加了特征列的DataFrame
        """
        close = df["close"]
        volume = df["volume"]

        # 收益率特征
        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)
        df["return_10d"] = close.pct_change(10)

        # 波动率特征
        df["volatility_10d"] = Indicators.volatility(close, 10)
        df["volatility_20d"] = Indicators.volatility(close, 20)

        # 量比
        df["volume_ratio"] = Indicators.volume_ratio(volume, 20)

        # 均线偏离度
        ma5 = Indicators.moving_average(close, 5)
        ma20 = Indicators.moving_average(close, 20)
        df["ma5_deviation"] = (close - ma5) / (ma5 + 1e-10)
        df["ma20_deviation"] = (close - ma20) / (ma20 + 1e-10)

        # RSI
        df["rsi_14"] = Indicators.rsi(close, 14)

        # MACD
        _, _, hist = Indicators.macd(close)
        df["macd_hist"] = hist

        return df

    def train(self, df: pd.DataFrame):
        """
        训练ML模型

        标签定义：次日收益率 > 0 为 1（涨），否则为 0（跌）
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import classification_report

        df = self.prepare_features(df.copy())

        # 构建标签：次日收益方向
        df["target"] = (df["close"].shift(-1) / df["close"] - 1 > 0).astype(int)
        df = df.dropna()

        if len(df) < 100:
            logging.warning("数据量不足100条，跳过ML训练")
            self.is_trained = False
            return

        # 按比例划分训练集/测试集（时间序列不可shuffle）
        split_idx = int(len(df) * self.ml_cfg["train_ratio"])
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[self.feature_cols]
        y_train = train_df["target"]
        X_test = test_df[self.feature_cols]
        y_test = test_df["target"]

        # 选择模型
        if self.ml_cfg["model"] == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # 评估
        y_pred = self.model.predict(X_test)
        acc = (y_pred == y_test.values).mean()
        logging.info(f"[ML] 模型训练完成, 测试集准确率: {acc:.2%}")
        logging.info(f"\n{classification_report(y_test, y_pred, target_names=['跌', '涨'])}")

        # 特征重要性
        importance = pd.Series(self.model.feature_importances_, index=self.feature_cols)
        importance = importance.sort_values(ascending=False)
        logging.info(f"[ML] 特征重要性:\n{importance.to_string()}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        对最新数据做预测

        返回: 预测概率 Series (1=看涨概率)
        """
        if not self.is_trained or self.model is None:
            return pd.Series([0.5] * len(df), index=df.index)

        df = self.prepare_features(df.copy())
        features = df[self.feature_cols].fillna(0)

        # predict_proba 返回 [P(跌), P(涨)]
        proba = self.model.predict_proba(features)[:, 1]
        return pd.Series(proba, index=df.index)


# ============================================================
# 第五部分：策略引擎
# ============================================================


class StrategyEngine:
    """
    策略引擎：双均线交叉 + 成交量过滤 + ML预测

    信号逻辑:
      买入信号 = 快线上穿慢线 AND (成交量放大 OR ML看涨概率>0.55)
      卖出信号 = 快线下穿慢线 OR (ML看跌概率>0.6)

    三重确认机制降低假信号
    """

    def __init__(self, config: dict):
        self.cfg = config["strategy"]
        self.ma_cfg = self.cfg["ma"]
        self.vol_cfg = self.cfg["volume"]
        self.ind = Indicators()
        self.ml = MLPredictor(config) if self.cfg["ml"]["enabled"] else None

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算交易信号

        参数:
            df: 包含OHLCV的DataFrame
        返回:
            添加了 signal 列的DataFrame (1=买, -1=卖, 0=持有)
        """
        df = df.copy()

        # 1. 计算均线
        df["ma_fast"] = self.ind.moving_average(df["close"], self.ma_cfg["fast_period"])
        df["ma_slow"] = self.ind.moving_average(df["close"], self.ma_cfg["slow_period"])

        # 2. 均线交叉信号
        df["ma_diff"] = df["ma_fast"] - df["ma_slow"]
        df["ma_diff_prev"] = df["ma_diff"].shift(1)
        # 金叉：diff从负变正；死叉：diff从正变负
        df["golden_cross"] = (df["ma_diff"] > 0) & (df["ma_diff_prev"] <= 0)
        df["death_cross"] = (df["ma_diff"] < 0) & (df["ma_diff_prev"] >= 0)

        # 3. 成交量过滤
        if self.vol_cfg["enabled"]:
            df["vol_avg"] = self.ind.moving_average(df["volume"], self.vol_cfg["avg_period"])
            df["vol_surge"] = df["volume"] > (df["vol_avg"] * self.vol_cfg["multiplier"])
        else:
            df["vol_surge"] = True

        # 4. ML预测
        if self.ml is not None:
            df["ml_prob"] = self.ml.predict(df)
        else:
            df["ml_prob"] = 0.5

        # 5. 综合信号
        df["signal"] = Signal.HOLD.value

        # 买入条件：金叉 AND (放量 OR ML看涨)
        buy_mask = df["golden_cross"] & (df["vol_surge"] | (df["ml_prob"] > 0.55))
        df.loc[buy_mask, "signal"] = Signal.BUY.value

        # 卖出条件：死叉 OR ML强烈看跌
        sell_mask = df["death_cross"] | (df["ml_prob"] < 0.4)
        df.loc[sell_mask, "signal"] = Signal.SELL.value

        # 同一天买卖冲突时，优先卖出（风控优先）
        conflict_mask = buy_mask & sell_mask
        df.loc[conflict_mask, "signal"] = Signal.SELL.value

        buy_count = (df["signal"] == Signal.BUY.value).sum()
        sell_count = (df["signal"] == Signal.SELL.value).sum()
        logging.info(f"[策略] 信号统计 - 买入: {buy_count}, 卖出: {sell_count}, 持有: {len(df) - buy_count - sell_count}")

        return df

    def train_ml_model(self, df: pd.DataFrame):
        """训练ML模型"""
        if self.ml is not None:
            self.ml.train(df)


# ============================================================
# 第六部分：风控模块
# ============================================================


class RiskManager:
    """
    风控模块

    功能:
    1. 仓位管理 - 单股上限、总仓位上限
    2. 止损止盈 - 基于买入成本的百分比
    3. 日亏损限额 - 当日累计亏损超限则停止交易
    4. 最大回撤控制 - 触发后全平仓
    """

    def __init__(self, config: dict, initial_capital: float):
        self.risk_cfg = config["risk"]
        self.pos_cfg = self.risk_cfg["position"]
        self.sl_cfg = self.risk_cfg["stop_loss"]
        self.tp_cfg = self.risk_cfg["take_profit"]
        self.dl_cfg = self.risk_cfg["daily_loss"]
        self.md_cfg = self.risk_cfg["max_drawdown"]
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.daily_pnl = 0.0
        self.current_date = ""

    def reset_daily(self, date: str):
        """每日重置日亏损计数"""
        if date != self.current_date:
            self.daily_pnl = 0.0
            self.current_date = date

    def record_trade_pnl(self, pnl: float):
        """记录交易盈亏"""
        self.daily_pnl += pnl

    def check_position_limit(self, stock_code: str, buy_amount: float, total_equity: float, positions: dict) -> Tuple[bool, str]:
        """
        检查仓位限制

        参数:
            buy_amount: 计划买入金额
            total_equity: 总资产
            positions: 当前持仓字典
        返回:
            (是否允许, 原因)
        """
        # 单股仓位检查
        max_single = total_equity * self.pos_cfg["max_position_pct"]
        current_pos = positions.get(stock_code)
        current_value = current_pos.market_value if current_pos else 0
        if current_value + buy_amount > max_single:
            return False, f"单股仓位超限: 当前{current_value:.0f} + 买入{buy_amount:.0f} > 上限{max_single:.0f}"

        # 总仓位检查
        total_position_value = sum(p.market_value for p in positions.values() if p.quantity > 0)
        max_total = total_equity * self.pos_cfg["total_position_pct"]
        if total_position_value + buy_amount > max_total:
            return False, f"总仓位超限: {total_position_value:.0f} + {buy_amount:.0f} > 上限{max_total:.0f}"

        return True, "OK"

    def check_stop_loss_take_profit(self, position: Position) -> Optional[str]:
        """
        检查止损止盈

        返回:
            None = 无需操作, "stop_loss" = 触发止损, "take_profit" = 触发止盈
        """
        if position.quantity <= 0:
            return None

        price = position.current_price
        cost = position.avg_cost

        # 止损
        if self.sl_cfg["enabled"]:
            loss_pct = (cost - price) / cost
            if loss_pct >= self.sl_cfg["pct"]:
                return "stop_loss"

        # 止盈
        if self.tp_cfg["enabled"]:
            gain_pct = (price - cost) / cost
            if gain_pct >= self.tp_cfg["pct"]:
                return "take_profit"

        return None

    def check_daily_loss_limit(self, total_equity: float) -> Tuple[bool, str]:
        """检查日亏损限额"""
        if not self.dl_cfg["enabled"]:
            return True, "OK"

        max_daily_loss = self.initial_capital * self.dl_cfg["max_daily_loss_pct"]
        if self.daily_pnl < -max_daily_loss:
            return False, f"日亏损超限: {self.daily_pnl:.0f} < -{max_daily_loss:.0f}"
        return True, "OK"

    def check_max_drawdown(self, total_equity: float) -> Tuple[bool, str]:
        """检查最大回撤"""
        if not self.md_cfg["enabled"]:
            return True, "OK"

        # 更新历史最高净值
        self.peak_equity = max(self.peak_equity, total_equity)
        drawdown = (self.peak_equity - total_equity) / self.peak_equity

        if drawdown >= self.md_cfg["pct"]:
            return False, f"最大回撤触发: {drawdown:.2%} >= {self.md_cfg['pct']:.2%}"
        return True, "OK"

    def calculate_position_size(self, price: float, total_equity: float) -> int:
        """
        计算建议买入股数

        使用等权分配：将可用资金平均分配到标的数，
        此处简化为总资产的 max_position_pct 用于单只股票
        """
        max_amount = total_equity * self.pos_cfg["max_position_pct"]
        shares = int(max_amount / price / 100) * 100  # A股100股整数倍
        return max(shares, 0)


# ============================================================
# 第七部分：回测引擎
# ============================================================


@dataclass
class BacktestResult:
    """回测结果汇总"""

    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_date: str = ""
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_records: List[TradeRecord] = field(default_factory=list)


class BacktestEngine:
    """
    回测引擎

    模拟真实的A股交易环境:
    - T+1 限制（当日买入次日才能卖出）
    - 佣金 + 印花税 + 滑点
    - 100股整数倍
    """

    def __init__(self, config: dict):
        self.bt_cfg = config["backtest"]
        self.initial_capital = self.bt_cfg["initial_capital"]
        self.commission_rate = self.bt_cfg["commission_rate"]
        self.stamp_tax_rate = self.bt_cfg["stamp_tax_rate"]
        self.slippage_pct = self.bt_cfg["slippage_pct"]
        self.min_commission = 5.0  # 最低佣金5元

        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_records: List[TradeRecord] = []
        self.equity_history: List[dict] = []

    def _calc_commission(self, amount: float, side: OrderSide) -> float:
        """计算佣金"""
        commission = amount * self.commission_rate
        return max(commission, self.min_commission)

    def _calc_tax(self, amount: float, side: OrderSide) -> float:
        """计算印花税（仅卖出时收取）"""
        if side == OrderSide.SELL:
            return amount * self.stamp_tax_rate
        return 0.0

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """模拟滑点"""
        if side == OrderSide.BUY:
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)

    def execute_order(self, stock_code: str, side: OrderSide, price: float, quantity: int, date: str, reason: str = "") -> bool:
        """
        执行订单

        返回: 是否执行成功
        """
        exec_price = self._apply_slippage(price, side)
        amount = exec_price * quantity
        commission = self._calc_commission(amount, side)
        tax = self._calc_tax(amount, side)
        total_cost = commission + tax

        if side == OrderSide.BUY:
            # 买入：检查资金是否充足
            if amount + total_cost > self.cash:
                logging.debug(f"[回测] 资金不足: 需要 {amount + total_cost:.0f}, 可用 {self.cash:.0f}")
                return False

            self.cash -= (amount + total_cost)

            if stock_code in self.positions:
                pos = self.positions[stock_code]
                # 加权平均成本
                total_qty = pos.quantity + quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + exec_price * quantity) / total_qty
                pos.quantity = total_qty
            else:
                self.positions[stock_code] = Position(
                    stock_code=stock_code, quantity=quantity, avg_cost=exec_price
                )

            self.positions[stock_code].stop_loss_price = exec_price * 0.95
            self.positions[stock_code].take_profit_price = exec_price * 1.10

        elif side == OrderSide.SELL:
            if stock_code not in self.positions or self.positions[stock_code].quantity < quantity:
                logging.debug(f"[回测] 持仓不足: {stock_code}")
                return False

            pos = self.positions[stock_code]
            pnl = (exec_price - pos.avg_cost) * quantity - total_cost
            self.cash += (amount - total_cost)

            pos.quantity -= quantity
            if pos.quantity == 0:
                pos.avg_cost = 0.0

        # 记录成交
        self.trade_records.append(
            TradeRecord(
                stock_code=stock_code,
                side=side.value,
                price=exec_price,
                quantity=quantity,
                amount=amount,
                commission=commission,
                tax=tax,
                timestamp=date,
                reason=reason,
            )
        )
        return True

    def snapshot_equity(self, date: str, prices: Dict[str, float]):
        """记录每日权益快照"""
        # 更新持仓市价
        position_value = 0
        for code, pos in self.positions.items():
            if code in prices:
                pos.update_price(prices[code])
            position_value += pos.market_value

        total_equity = self.cash + position_value
        self.equity_history.append(
            {
                "date": date,
                "cash": self.cash,
                "position_value": position_value,
                "total_equity": total_equity,
            }
        )

    def run(self, stock_data: Dict[str, pd.DataFrame], strategy: StrategyEngine, risk_mgr: RiskManager) -> BacktestResult:
        """
        执行回测主循环

        参数:
            stock_data: {stock_code: DataFrame} 各股票的行情数据
            strategy: 策略引擎
            risk_mgr: 风控管理器
        """
        # 对每只股票计算信号
        signal_data = {}
        for code, df in stock_data.items():
            signal_data[code] = strategy.compute_signals(df)

        # 收集所有交易日期并排序
        all_dates = set()
        for df in signal_data.values():
            all_dates.update(df["date"].dt.strftime("%Y-%m-%d").tolist())
        all_dates = sorted(all_dates)

        logging.info(f"[回测] 开始回测, 日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 个交易日")

        # 训练ML模型（使用前70%数据训练）
        if strategy.ml is not None:
            for code, df in signal_data.items():
                strategy.train_ml_model(df)

        # 逐日回测
        for date_str in all_dates:
            date_dt = pd.Timestamp(date_str)
            risk_mgr.reset_daily(date_str)
            daily_prices = {}

            for code, df in signal_data.items():
                day_row = df[df["date"] == date_dt]
                if day_row.empty:
                    continue
                row = day_row.iloc[0]
                price = row["close"]
                signal = row["signal"]
                daily_prices[code] = price

                # 更新持仓市价
                if code in self.positions:
                    self.positions[code].update_price(price)

                total_equity = self.cash + sum(p.market_value for p in self.positions.values())

                # === 风控检查 ===
                # 1. 最大回撤检查
                ok, msg = risk_mgr.check_max_drawdown(total_equity)
                if not ok:
                    logging.warning(f"[风控] {msg} -> 全部清仓")
                    for c, p in self.positions.items():
                        if p.quantity > 0:
                            self.execute_order(c, OrderSide.SELL, p.current_price, p.quantity, date_str, reason="最大回撤清仓")
                    break

                # 2. 日亏损检查
                ok, msg = risk_mgr.check_daily_loss_limit(total_equity)
                if not ok:
                    logging.warning(f"[风控] {msg} -> 今日停止交易")
                    continue

                # 3. 止损止盈检查
                if code in self.positions:
                    sl_tp = risk_mgr.check_stop_loss_take_profit(self.positions[code])
                    if sl_tp == "stop_loss":
                        pos = self.positions[code]
                        self.execute_order(code, OrderSide.SELL, price, pos.quantity, date_str, reason="止损")
                        risk_mgr.record_trade_pnl((price - pos.avg_cost) * pos.quantity)
                        continue
                    elif sl_tp == "take_profit":
                        pos = self.positions[code]
                        self.execute_order(code, OrderSide.SELL, price, pos.quantity, date_str, reason="止盈")
                        risk_mgr.record_trade_pnl((price - pos.avg_cost) * pos.quantity)
                        continue

                # === 策略信号执行 ===
                if signal == Signal.BUY.value:
                    quantity = risk_mgr.calculate_position_size(price, total_equity)
                    if quantity > 0:
                        buy_amount = price * quantity
                        ok, msg = risk_mgr.check_position_limit(code, buy_amount, total_equity, self.positions)
                        if ok:
                            self.execute_order(code, OrderSide.BUY, price, quantity, date_str, reason="策略买入")
                        else:
                            logging.debug(f"[风控] 买入被拒: {msg}")

                elif signal == Signal.SELL.value:
                    if code in self.positions and self.positions[code].quantity > 0:
                        pos = self.positions[code]
                        pnl = (price - pos.avg_cost) * pos.quantity
                        self.execute_order(code, OrderSide.SELL, price, pos.quantity, date_str, reason="策略卖出")
                        risk_mgr.record_trade_pnl(pnl)

            # 记录当日权益
            self.snapshot_equity(date_str, daily_prices)

        # 生成回测报告
        return self._generate_result()

    def _generate_result(self) -> BacktestResult:
        """生成回测结果统计"""
        result = BacktestResult()
        result.trade_records = self.trade_records
        result.total_trades = len(self.trade_records)

        if not self.equity_history:
            return result

        eq = pd.DataFrame(self.equity_history)
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.set_index("date")
        result.equity_curve = eq

        # 总收益率
        final_equity = eq["total_equity"].iloc[-1]
        result.total_return = (final_equity - self.initial_capital) / self.initial_capital

        # 年化收益率
        trading_days = len(eq)
        years = trading_days / 252
        if years > 0:
            result.annual_return = (1 + result.total_return) ** (1 / years) - 1

        # 夏普比率（假设无风险利率2%）
        daily_returns = eq["total_equity"].pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess_return = daily_returns.mean() - 0.02 / 252
            result.sharpe_ratio = excess_return / daily_returns.std() * np.sqrt(252)

        # 最大回撤
        cummax = eq["total_equity"].cummax()
        drawdown = (cummax - eq["total_equity"]) / cummax
        result.max_drawdown = drawdown.max()
        if drawdown.max() > 0:
            result.max_drawdown_date = str(drawdown.idxmax().date())

        # 胜率
        sell_trades = [t for t in self.trade_records if t.side == "sell"]
        if sell_trades:
            wins = []
            losses = []
            for t in sell_trades:
                # 找对应的买入记录
                buy_price = None
                for bt in self.trade_records:
                    if bt.side == "buy" and bt.stock_code == t.stock_code and bt.timestamp <= t.timestamp:
                        buy_price = bt.price
                if buy_price:
                    pnl = (t.price - buy_price) * t.quantity - t.commission - t.tax
                    if pnl > 0:
                        wins.append(pnl)
                    else:
                        losses.append(pnl)

            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            if len(wins) + len(losses) > 0:
                result.win_rate = len(wins) / (len(wins) + len(losses))
            if wins:
                result.avg_win = np.mean(wins)
            if losses:
                result.avg_loss = np.mean(losses)
            if losses and sum(losses) != 0:
                result.profit_factor = abs(sum(wins) / sum(losses))

        return result


# ============================================================
# 第八部分：实盘执行框架（模拟 + 券商API预留）
# ============================================================


class LiveTrader:
    """
    实盘交易框架

    当前实现：模拟交易（纸交易）
    预留接口：对接券商API（东方财富、华泰等）
    """

    def __init__(self, config: dict):
        self.live_cfg = config.get("live_trading", {})
        self.broker = self.live_cfg.get("broker", "simulation")
        self.enabled = self.live_cfg.get("enabled", False)
        self.data_fetcher = DataFetcher(config)
        self.risk_mgr = RiskManager(config, 1000000)
        self.positions: Dict[str, Position] = {}
        self.cash = 1000000.0
        self.order_history: List[Order] = []

    def connect_broker(self) -> bool:
        """
        连接券商API（预留接口）

        实际对接时需实现:
        - 东方财富: EMQuantAPI
        - 华泰: MATIC
        - 通达信: pytdx
        """
        if self.broker == "simulation":
            logging.info("[实盘] 使用模拟交易模式")
            return True

        # TODO: 对接真实券商API
        # if self.broker == "eastmoney":
        #     from emquantapi import c
        #     c.start()
        # elif self.broker == "htsc":
        #     import matic
        #     ...
        logging.warning(f"[实盘] 券商 {self.broker} 接口未实现，回退到模拟模式")
        return True

    def place_order(self, stock_code: str, side: OrderSide, quantity: int, reason: str = "") -> Optional[Order]:
        """
        下单

        模拟模式：直接以最新价成交
        实盘模式：发送至券商API
        """
        quote = self.data_fetcher.get_realtime_quote(stock_code)
        if not quote:
            logging.error(f"[实盘] 无法获取 {stock_code} 行情")
            return None

        price = quote["price"]

        order = Order(
            stock_code=stock_code,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason=reason,
        )

        if self.broker == "simulation":
            # 模拟成交
            order.status = "filled"
            self.order_history.append(order)
            logging.info(f"[模拟成交] {side.value} {stock_code} x{quantity} @ {price:.2f} | {reason}")
        else:
            # TODO: 发送至券商API
            order.status = "pending"
            logging.info(f"[实盘下单] {side.value} {stock_code} x{quantity} @ {price:.2f}")

        return order

    def check_and_execute(self, strategy: StrategyEngine, config: dict):
        """
        检查信号并执行交易（实盘主循环调用）

        这个方法在实盘中被定时调度（如每分钟/每日调用一次）
        """
        stock_pool = config["data"]["stock_pool"]
        start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        for code in stock_pool:
            df = self.data_fetcher.get_daily_data(code, start_date, end_date)
            if df.empty:
                continue

            signal_df = strategy.compute_signals(df)
            latest = signal_df.iloc[-1]
            signal = latest["signal"]

            if signal == Signal.BUY.value:
                quantity = self.risk_mgr.calculate_position_size(latest["close"], self.cash)
                if quantity > 0:
                    self.place_order(code, OrderSide.BUY, quantity, reason="策略信号买入")

            elif signal == Signal.SELL.value:
                if code in self.positions and self.positions[code].quantity > 0:
                    qty = self.positions[code].quantity
                    self.place_order(code, OrderSide.SELL, qty, reason="策略信号卖出")


# ============================================================
# 第九部分：报告与可视化
# ============================================================


class Reporter:
    """回测报告与可视化"""

    @staticmethod
    def print_summary(result: BacktestResult):
        """打印回测摘要到控制台"""
        print("\n" + "=" * 60)
        print("  量化交易系统 - 回测报告")
        print("  风险警告: 回测结果不代表未来收益，仅供学习参考")
        print("=" * 60)

        data = [
            ["总收益率", f"{result.total_return:.2%}"],
            ["年化收益率", f"{result.annual_return:.2%}"],
            ["夏普比率", f"{result.sharpe_ratio:.2f}"],
            ["最大回撤", f"{result.max_drawdown:.2%}"],
            ["最大回撤日期", result.max_drawdown_date],
            ["总交易次数", result.total_trades],
            ["胜率", f"{result.win_rate:.2%}"],
            ["盈利因子", f"{result.profit_factor:.2f}"],
            ["盈利笔数", result.winning_trades],
            ["亏损笔数", result.losing_trades],
            ["平均盈利", f"{result.avg_win:.0f}"],
            ["平均亏损", f"{result.avg_loss:.0f}"],
        ]
        print(tabulate(data, headers=["指标", "数值"], tablefmt="grid"))
        print()

        # 打印最近10笔交易
        if result.trade_records:
            recent = result.trade_records[-10:]
            trade_data = [
                [t.timestamp, t.stock_code, t.side, f"{t.price:.2f}", t.quantity, f"{t.amount:.0f}", t.reason]
                for t in recent
            ]
            print("最近交易记录:")
            print(tabulate(trade_data, headers=["日期", "代码", "方向", "价格", "数量", "金额", "原因"], tablefmt="grid"))

    @staticmethod
    def plot_results(result: BacktestResult, save_path: str = None):
        """绘制回测结果图表"""
        if result.equity_curve.empty:
            logging.warning("无权益曲线数据，跳过绘图")
            return

        eq = result.equity_curve

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.suptitle("Quantitative Trading System - Backtest Report\n(Risk Warning: Past performance does not guarantee future results)", fontsize=14)

        # 图1: 权益曲线
        ax1 = axes[0]
        ax1.plot(eq.index, eq["total_equity"], label="Total Equity", color="#2196F3", linewidth=1.5)
        ax1.axhline(y=result.equity_curve["total_equity"].iloc[0], color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
        ax1.fill_between(eq.index, eq["total_equity"], eq["total_equity"].iloc[0], alpha=0.1, color="#2196F3")
        ax1.set_ylabel("Equity (CNY)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Equity Curve")

        # 图2: 回撤曲线
        ax2 = axes[1]
        cummax = eq["total_equity"].cummax()
        drawdown = (cummax - eq["total_equity"]) / cummax * 100
        ax2.fill_between(eq.index, drawdown, 0, color="#F44336", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown")
        ax2.grid(True, alpha=0.3)

        # 图3: 仓位占比
        ax3 = axes[2]
        position_pct = eq["position_value"] / eq["total_equity"] * 100
        ax3.fill_between(eq.index, position_pct, 0, color="#4CAF50", alpha=0.4)
        ax3.set_ylabel("Position (%)")
        ax3.set_title("Position Ratio")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logging.info(f"[报告] 图表已保存: {save_path}")
        else:
            plt.savefig("backtest_report.png", dpi=150, bbox_inches="tight")
            logging.info("[报告] 图表已保存: backtest_report.png")

        plt.close()

    @staticmethod
    def save_trade_log(trades: List[TradeRecord], filepath: str = "trade_log.csv"):
        """保存交易记录到CSV"""
        if not trades:
            return
        df = pd.DataFrame(
            [
                {
                    "日期": t.timestamp,
                    "股票代码": t.stock_code,
                    "方向": t.side,
                    "价格": f"{t.price:.2f}",
                    "数量": t.quantity,
                    "金额": f"{t.amount:.0f}",
                    "佣金": f"{t.commission:.2f}",
                    "印花税": f"{t.tax:.2f}",
                    "原因": t.reason,
                }
                for t in trades
            ]
        )
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logging.info(f"[报告] 交易记录已保存: {filepath}")


# ============================================================
# 第十部分：主入口
# ============================================================


def setup_logging(level: str = "INFO"):
    """配置日志"""
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=log_format, datefmt="%Y-%m-%d %H:%M:%S")


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="A股程序化量化交易系统")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest", help="运行模式: backtest=回测, live=实盘")
    parser.add_argument("--config", default="strategy.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    logging.info("=" * 50)
    logging.info("  A股程序化量化交易系统启动")
    logging.info("  灵感来源: 梁文锋 / 幻方量化 早期策略思路")
    logging.info("  免责声明: 仅供学习，不构成投资建议")
    logging.info("=" * 50)

    # 2. 初始化数据获取
    fetcher = DataFetcher(config)

    # 3. 获取股票数据
    stock_pool = config["data"]["stock_pool"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    stock_data = {}
    for code in stock_pool:
        df = fetcher.get_daily_data(code, start_date, end_date)
        if not df.empty:
            stock_data[code] = df
        time.sleep(0.5)  # 避免请求过快

    if not stock_data:
        logging.error("未能获取任何股票数据，请检查网络或配置")
        sys.exit(1)

    logging.info(f"成功获取 {len(stock_data)} 只股票数据")

    # 4. 初始化策略引擎并训练ML模型
    strategy = StrategyEngine(config)

    # 5. 根据模式运行
    if args.mode == "backtest":
        logging.info("===== 回测模式 =====")

        # 初始化回测引擎和风控
        bt_engine = BacktestEngine(config)
        risk_mgr = RiskManager(config, config["backtest"]["initial_capital"])

        # 执行回测
        result = bt_engine.run(stock_data, strategy, risk_mgr)

        # 输出报告
        Reporter.print_summary(result)

        report_dir = config.get("logging", {}).get("report_dir", "./reports")
        os.makedirs(report_dir, exist_ok=True)

        Reporter.plot_results(result, save_path=os.path.join(report_dir, "backtest_report.png"))
        Reporter.save_trade_log(result.trade_records, filepath=os.path.join(report_dir, "trade_log.csv"))

        # 保存权益曲线
        if not result.equity_curve.empty:
            eq_path = os.path.join(report_dir, "equity_curve.csv")
            result.equity_curve.to_csv(eq_path)
            logging.info(f"[报告] 权益曲线已保存: {eq_path}")

    elif args.mode == "live":
        logging.info("===== 实盘模式 =====")
        logging.warning("⚠️  实盘交易有真实资金风险，请确保已充分测试！")

        trader = LiveTrader(config)
        if trader.connect_broker():
            trader.check_and_execute(strategy, config)
        else:
            logging.error("券商连接失败")
            sys.exit(1)

    logging.info("程序运行结束")


if __name__ == "__main__":
    main()
