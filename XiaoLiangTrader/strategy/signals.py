"""
信号定义 — 所有策略共用的数据结构
==================================
Action: 买/卖/持有
Signal: 一次交易信号的完整信息
"""

from enum import Enum
from dataclasses import dataclass


class Action(Enum):
    """交易动作"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    交易信号 — 策略产出、执行器消费

    Attributes:
        action: 买/卖/持有
        symbol: 股票代码
        price: 当前价格
        size: 建议股数（0 表示由执行器决定）
        reason: 信号原因（方便日志和调试）
        ma_fast: 短期均线值
        ma_slow: 长期均线值
        volume_ratio: 当前成交量 / 均量
        ml_score: ML 模型的置信度（0~1，-1 表示未使用）
        llm_opinion: LLM 的建议文本
    """
    action: Action
    symbol: str
    price: float
    size: int = 0
    reason: str = ""
    ma_fast: float = 0.0
    ma_slow: float = 0.0
    volume_ratio: float = 0.0
    ml_score: float = -1.0
    llm_opinion: str = ""
