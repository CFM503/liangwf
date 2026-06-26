"""
ML 增强策略 — 双均线 + 传统机器学习过滤
==========================================
核心思想：
  双均线给方向，ML 给置信度。
  只有当模型认为"大概率涨"时才买入。
  ML 不独立决策，只是双均线的过滤器。

这样即使模型不准，双均线的止损还能兜底。
"""

import pandas as pd
from .signals import Signal, Action
from .dual_ma import DualMAStrategy
from utils.logger import get_logger

log = get_logger("xlt.ml_strategy")


class MLEnhancedStrategy:
    """
    ML 增强版策略

    流程：
    1. 双均线策略给出基础信号
    2. ML 模型（LightGBM/XGBoost/sklearn）预测上涨概率
    3. 只有当概率 > 阈值时，才执行买入
    4. 卖出信号不受 ML 影响（止损必须执行）
    """

    def __init__(
        self,
        base_strategy: DualMAStrategy,
        ml_confidence: float = 0.6,
        ml_predictor=None,
    ):
        self.base = base_strategy
        self.ml_confidence = ml_confidence
        self.predictor = ml_predictor  # ml_model.predictor.MLPredictor 实例

    def get_latest_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        position_size: int = 0,
        buy_price: float = 0.0,
        max_price: float = 0.0,
    ) -> Signal:
        """
        获取信号（ML 增强版）

        只有 BUY 信号会经过 ML 过滤。
        SELL / HOLD 原样返回。
        """
        signal = self.base.get_latest_signal(
            df, symbol, position_size, buy_price, max_price
        )

        # 卖出信号不过滤（止损止盈必须执行）
        if signal.action != Action.BUY:
            return signal

        # 无 ML 模型时直接返回
        if self.predictor is None:
            return signal

        # 叠加 ML 过滤
        try:
            prob = self.predictor.predict_proba(df)
            signal.ml_score = prob

            if prob < self.ml_confidence:
                log.info(f"[ML] {symbol} 概率 {prob:.2f} < {self.ml_confidence:.2f}，放弃")
                return Signal(Action.HOLD, symbol, signal.price, 0,
                              f"ML过滤 ({prob:.2f})",
                              signal.ma_fast, signal.ma_slow, signal.volume_ratio,
                              ml_score=prob)
            else:
                log.info(f"[ML] {symbol} 概率 {prob:.2f} ✓")
                signal.reason += f" [ML:{prob:.2f}]"
        except Exception as e:
            log.warning(f"[ML] {symbol} 预测失败: {e}，使用纯策略信号")

        return signal
