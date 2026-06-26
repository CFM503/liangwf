"""
信号定义 — 所有策略共用的数据结构
==================================
"""

from enum import Enum
from dataclasses import dataclass


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    交易信号

    策略产出 → 执行器消费，中间经过风控校验。
    """
    action: Action
    symbol: str
    price: float
    size: int = 0
    reason: str = ""
    ma_fast: float = 0.0
    ma_slow: float = 0.0
    volume_ratio: float = 0.0
    ml_score: float = -1.0  # ML 置信度（-1 表示未使用）
