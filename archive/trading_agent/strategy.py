"""
策略引擎 — 纯 pandas 信号计算（不依赖 backtrader 运行时）
复用 strategy_dual_ma.py 的逻辑，输出结构化信号
"""

from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from .logger import log


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    action: Action
    symbol: str
    price: float
    size: int  # 股数，0 表示仅信号无具体数量
    reason: str
    ma_fast: float = 0.0
    ma_slow: float = 0.0
    volume_ratio: float = 0.0


class DualMAStrategy:
    """
    双均线交叉策略（信号计算版）

    与 strategy_dual_ma.py 逻辑一致，但不依赖 backtrader，
    直接用 pandas 计算，适合 Agent 实时调用。
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 20,
        vol_period: int = 20,
        vol_mult: float = 1.5,
        stop_loss: float = 0.08,
        take_profit: float = 0.15,
        trailing_pct: float = 0.05,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_pct = trailing_pct

    def compute_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        计算完整信号序列（用于回测）

        Args:
            df: OHLCV DataFrame
            symbol: 股票代码

        Returns:
            DataFrame 附加列: ma_fast, ma_slow, vol_ma, signal
        """
        df = df.copy()
        df["ma_fast"] = df["close"].rolling(self.fast_period).mean()
        df["ma_slow"] = df["close"].rolling(self.slow_period).mean()
        df["vol_ma"] = df["volume"].rolling(self.vol_period).mean()

        # 金叉 / 死叉
        cross = df["ma_fast"] - df["ma_slow"]
        df["cross"] = np.sign(cross)
        df["golden_cross"] = (df["cross"] > 0) & (df["cross"].shift(1) <= 0)
        df["death_cross"] = (df["cross"] < 0) & (df["cross"].shift(1) >= 0)

        # 成交量过滤
        df["vol_ok"] = df["volume"] > df["vol_ma"] * self.vol_mult

        # 信号
        df["signal"] = Action.HOLD
        df.loc[df["golden_cross"] & df["vol_ok"], "signal"] = Action.BUY
        df.loc[df["death_cross"], "signal"] = Action.SELL

        return df

    def get_latest_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        position_size: int = 0,
        buy_price: float = 0.0,
        max_price: float = 0.0,
    ) -> Signal:
        """
        获取最新一根 K 线的交易信号（实时/每日调用）

        Args:
            df: 至少 slow_period + 2 根 K 线
            symbol: 股票代码
            position_size: 当前持仓股数（0 表示空仓）
            buy_price: 买入价（持仓时必填）
            max_price: 持仓期最高价（持仓时必填）

        Returns:
            Signal
        """
        if len(df) < self.slow_period + 2:
            return Signal(Action.HOLD, symbol, 0, 0, "数据不足")

        df = self.compute_signals(df, symbol)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price = latest["close"]

        # ── 持仓中：检查卖出条件 ──
        if position_size > 0 and buy_price > 0:
            # 止损
            if price <= buy_price * (1 - self.stop_loss):
                return Signal(
                    Action.SELL, symbol, price, position_size,
                    f"止损 {self.stop_loss*100:.0f}%",
                    latest["ma_fast"], latest["ma_slow"],
                    latest["volume"] / latest["vol_ma"] if latest["vol_ma"] > 0 else 0,
                )

            # 跟踪止盈
            if max_price > buy_price * (1 + self.take_profit):
                if price <= max_price * (1 - self.trailing_pct):
                    return Signal(
                        Action.SELL, symbol, price, position_size,
                        f"跟踪止盈 (最高{max_price:.2f}回撤{self.trailing_pct*100:.0f}%)",
                        latest["ma_fast"], latest["ma_slow"],
                        latest["volume"] / latest["vol_ma"] if latest["vol_ma"] > 0 else 0,
                    )

            # 死叉
            if latest["death_cross"]:
                return Signal(
                    Action.SELL, symbol, price, position_size,
                    "死叉卖出",
                    latest["ma_fast"], latest["ma_slow"],
                    latest["volume"] / latest["vol_ma"] if latest["vol_ma"] > 0 else 0,
                )

            return Signal(
                Action.HOLD, symbol, price, position_size,
                "持仓中，无卖出信号",
                latest["ma_fast"], latest["ma_slow"],
                latest["volume"] / latest["vol_ma"] if latest["vol_ma"] > 0 else 0,
            )

        # ── 空仓：检查买入条件 ──
        if latest["golden_cross"] and latest["vol_ok"]:
            return Signal(
                Action.BUY, symbol, price, 0,  # size 由 executor 决定
                f"金叉买入 (量比{latest['volume']/latest['vol_ma']:.1f}x)",
                latest["ma_fast"], latest["ma_slow"],
                latest["volume"] / latest["vol_ma"] if latest["vol_ma"] > 0 else 0,
            )

        return Signal(
            Action.HOLD, symbol, price, 0,
            "无信号",
            latest["ma_fast"], latest["ma_slow"],
            latest["volume"] / latest["vol_ma"] if latest["vol_ma"] > 0 else 0,
        )
