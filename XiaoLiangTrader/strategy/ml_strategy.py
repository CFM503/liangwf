"""
ML 增强策略 — 在双均线基础上叠加 LightGBM 预测
================================================
核心思想：双均线给方向，ML 给置信度。
ML 不独立决策，只作为过滤器。
"""

import pandas as pd
import numpy as np
from .signals import Signal, Action
from .dual_ma import DualMAStrategy
from utils.logger import get_logger

log = get_logger("xlt.ml_strategy")


class MLEnhancedStrategy:
    """
    ML 增强版策略

    流程：
    1. 双均线策略给出基础信号
    2. LightGBM 模型给出"未来N天上涨概率"
    3. 只有当 ML 置信度 > 阈值时，才执行买入
    4. 可选：LLM 对当前行情给出"建议/不建议"

    这样即使 ML 模型不准，双均线的止损还能兜底。
    """

    def __init__(
        self,
        base_strategy: DualMAStrategy,
        ml_confidence: float = 0.6,
        lgb_model=None,
        llm_advisor=None,
    ):
        self.base = base_strategy
        self.ml_confidence = ml_confidence
        self.lgb_model = lgb_model      # ml_model.lgb_model.LightGBMModel 实例
        self.llm_advisor = llm_advisor  # ml_model.llm_advisor.LLMAdvisor 实例

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

        买入时叠加 ML 置信度过滤：
        - ML 预测上涨概率 < 阈值 → 降级为 HOLD
        - LLM 返回"不建议" → 降级为 HOLD（如果启用了 LLM）
        """
        # 1. 基础信号
        signal = self.base.get_latest_signal(
            df, symbol, position_size, buy_price, max_price
        )

        # 卖出信号不做过滤（止损/止盈必须执行）
        if signal.action == Action.SELL:
            return signal

        # HOLD 信号也不需要 ML 判断
        if signal.action == Action.HOLD:
            return signal

        # 2. 只有 BUY 信号才叠加 ML 过滤
        if signal.action == Action.BUY and self.lgb_model is not None:
            try:
                ml_prob = self.lgb_model.predict_proba(df)
                signal.ml_score = ml_prob

                if ml_prob < self.ml_confidence:
                    log.info(f"[ML] {symbol} 置信度 {ml_prob:.2f} < {self.ml_confidence:.2f}，放弃买入")
                    return Signal(Action.HOLD, symbol, signal.price, 0,
                                  f"ML过滤 (置信{ml_prob:.2f})",
                                  signal.ma_fast, signal.ma_slow, signal.volume_ratio,
                                  ml_score=ml_prob)
                else:
                    log.info(f"[ML] {symbol} 置信度 {ml_prob:.2f} ✓")
                    signal.reason += f" [ML:{ml_prob:.2f}]"
            except Exception as e:
                log.warning(f"[ML] {symbol} 预测失败: {e}，使用纯策略信号")

        # 3. 可选：LLM 辅助判断
        if signal.action == Action.BUY and self.llm_advisor is not None:
            try:
                opinion = self.llm_advisor.ask_opinion(df, symbol, signal)
                signal.llm_opinion = opinion.text

                if not opinion.should_buy:
                    log.info(f"[LLM] {symbol} 不建议买入: {opinion.text[:50]}")
                    return Signal(Action.HOLD, symbol, signal.price, 0,
                                  f"LLM否决 ({opinion.text[:30]})",
                                  signal.ma_fast, signal.ma_slow, signal.volume_ratio,
                                  ml_score=signal.ml_score, llm_opinion=opinion.text)
                else:
                    signal.reason += f" [LLM:OK]"
            except Exception as e:
                log.warning(f"[LLM] {symbol} 调用失败: {e}，跳过")

        return signal
