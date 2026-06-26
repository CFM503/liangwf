"""
批量选股引擎 — 这才是"校园股神"的核心玩法
============================================
不是盯一只股票，而是扫描整个市场，用模型打分排名。
每天收盘后跑一遍，输出"今日值得关注的 N 只股票"。

选股逻辑：
  1. 技术面打分（均线、量价、MACD、RSI）
  2. ML 模型打分（预测未来 3 天上涨概率）
  3. 综合排名 → 候选列表 + 建议仓位比例

这就是当年宿舍里每天下午 3 点跑的脚本。
"""

import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from data.fetcher import fetch_stock, STOCK_NAMES
from ml_model.predictor import MLPredictor
from ml_model.features import compute_features
from utils.logger import get_logger

log = get_logger("xlt.screener")


@dataclass
class StockScore:
    """一只股票的综合评分"""
    symbol: str
    name: str = ""
    price: float = 0.0

    # 技术面分项（0~100）
    ma_score: float = 0.0       # 均线得分
    vol_score: float = 0.0      # 成交量得分
    macd_score: float = 0.0     # MACD 得分
    rsi_score: float = 0.0      # RSI 得分
    trend_score: float = 0.0    # 趋势得分

    # 综合
    tech_score: float = 0.0     # 技术面综合（0~100）
    ml_prob: float = 0.0        # ML 预测上涨概率（0~1）
    final_score: float = 0.0    # 最终得分（0~100）

    # 信号
    signal: str = ""            # "BUY" / "HOLD" / "SELL"
    reason: str = ""
    position_pct: float = 0.0   # 建议仓位占比（%）


class StockScreener:
    """
    批量选股引擎

    用法:
        screener = StockScreener(ml_predictor=model)
        candidates = screener.scan(stock_list, start_date, end_date)
        screener.print_results(candidates)
    """

    def __init__(
        self,
        # 均线参数
        fast_period: int = 5,
        slow_period: int = 20,
        vol_period: int = 20,
        vol_mult: float = 1.5,
        # ML 模型（可选）
        ml_predictor: MLPredictor | None = None,
        ml_confidence: float = 0.6,
        # 选股参数
        min_tech_score: float = 40.0,   # 技术面最低分
        top_n: int = 10,                # 最终输出前 N 只
        max_position_pct: float = 20.0, # 单股最大仓位
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.ml_predictor = ml_predictor
        self.ml_confidence = ml_confidence
        self.min_tech_score = min_tech_score
        self.top_n = top_n
        self.max_position_pct = max_position_pct

    def scan(
        self,
        stock_codes: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[StockScore]:
        """
        批量扫描股票，返回按综合得分排序的候选列表。

        Args:
            stock_codes: 股票代码列表
            start_date: 数据起始日（默认90天前）
            end_date: 数据截止日（默认今天）

        Returns:
            list[StockScore]，按 final_score 降序
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        log.info(f"[选股] 开始扫描 {len(stock_codes)} 只股票...")
        t0 = time.time()

        results: list[StockScore] = []
        for i, code in enumerate(stock_codes):
            try:
                score = self._score_stock(code, start_date, end_date)
                if score is not None:
                    results.append(score)
            except Exception as e:
                log.warning(f"[选股] {code} 失败: {e}")

            # 进度（每 20 只打印一次）
            if (i + 1) % 20 == 0:
                log.info(f"[选股] 已扫描 {i+1}/{len(stock_codes)}...")

        # 按综合得分排序
        results.sort(key=lambda x: x.final_score, reverse=True)

        # 信号判定 + 仓位分配
        results = self._assign_signals(results)
        results = self._allocate_positions(results)

        elapsed = time.time() - t0
        buy_count = sum(1 for r in results if r.signal == "BUY")
        log.info(f"[选股] 扫描完成 | {elapsed:.1f}s | {len(results)} 只 | {buy_count} 只买入信号")

        return results[:self.top_n]

    def _score_stock(self, symbol: str, start_date: str, end_date: str) -> StockScore | None:
        """对单只股票打分"""
        df = fetch_stock(symbol, start_date, end_date, use_cache=True)
        if df.empty or len(df) < 65:
            return None

        score = StockScore(symbol=symbol, name=STOCK_NAMES.get(symbol, ""))
        score.price = df["close"].iloc[-1]

        # ── 1. 技术面打分 ──
        score.ma_score = self._score_ma(df)
        score.vol_score = self._score_volume(df)
        score.macd_score = self._score_macd(df)
        score.rsi_score = self._score_rsi(df)
        score.trend_score = self._score_trend(df)

        # 技术面综合（加权平均）
        score.tech_score = (
            score.ma_score * 0.30 +
            score.vol_score * 0.20 +
            score.macd_score * 0.20 +
            score.rsi_score * 0.15 +
            score.trend_score * 0.15
        )

        # ── 2. ML 预测 ──
        if self.ml_predictor is not None:
            try:
                score.ml_prob = self.ml_predictor.predict_proba(df)
            except Exception:
                score.ml_prob = 0.5

        # ── 3. 最终得分 ──
        if self.ml_predictor is not None:
            # 技术面 40% + ML 60%
            score.final_score = score.tech_score * 0.4 + score.ml_prob * 100 * 0.6
        else:
            score.final_score = score.tech_score

        return score

    # ──────────────────────────────────
    # 5 个分项打分函数（0~100）
    # ──────────────────────────────────

    def _score_ma(self, df: pd.DataFrame) -> float:
        """
        均线打分：
        - 金叉刚刚发生 → 高分
        - 多头排列（ma5 > ma10 > ma20）→ 加分
        - 死叉 → 0 分
        - 价格在均线下方 → 减分
        """
        close = df["close"]
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()

        last_close = close.iloc[-1]
        last_ma5 = ma5.iloc[-1]
        last_ma10 = ma10.iloc[-1]
        last_ma20 = ma20.iloc[-1]

        score = 50.0  # 基准分

        # 多头排列
        if last_ma5 > last_ma10 > last_ma20:
            score += 20
        elif last_ma5 < last_ma10 < last_ma20:
            score -= 20

        # 金叉（近 3 天内发生）
        diff = ma5 - ma20
        sign = np.sign(diff)
        for lookback in range(1, 4):
            if len(sign) > lookback and sign.iloc[-lookback] > 0 and sign.iloc[-lookback - 1] <= 0:
                score += 20
                break

        # 近 3 天死叉
        for lookback in range(1, 4):
            if len(sign) > lookback and sign.iloc[-lookback] < 0 and sign.iloc[-lookback - 1] >= 0:
                score -= 30
                break

        # 价格在均线之上
        if last_close > last_ma5:
            score += 5
        if last_close > last_ma20:
            score += 5

        return np.clip(score, 0, 100)

    def _score_volume(self, df: pd.DataFrame) -> float:
        """
        成交量打分：
        - 近 3 天放量（量比 > 1.5）+ 价格上涨 → 高分
        - 缩量下跌 → 低分
        - 天量（量比 > 3）→ 可能见顶，减分
        """
        vol = df["volume"]
        close = df["close"]
        vol_ma = vol.rolling(20).mean()

        last_vol_ratio = vol.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1
        price_chg = (close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2 else 0

        score = 50.0

        # 温和放量上涨（最佳）
        if 1.3 < last_vol_ratio < 3.0 and price_chg > 0:
            score += 30
        elif last_vol_ratio > 1.3 and price_chg > 0:
            score += 15

        # 缩量下跌
        if last_vol_ratio < 0.8 and price_chg < 0:
            score -= 20

        # 天量（可能见顶）
        if last_vol_ratio > 5.0:
            score -= 15

        # 连续 3 天放量
        recent_ratios = vol.iloc[-3:] / vol_ma.iloc[-3:]
        if (recent_ratios > 1.2).all():
            score += 10

        return np.clip(score, 0, 100)

    def _score_macd(self, df: pd.DataFrame) -> float:
        """
        MACD 打分：
        - 金叉 → 高分
        - 红柱放大 → 加分
        - 死叉 → 0 分
        """
        close = df["close"]
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd_hist = dif - dea

        score = 50.0

        # MACD 金叉（近 5 天）
        for lookback in range(1, 6):
            if len(dif) > lookback + 1:
                if dif.iloc[-lookback] > dea.iloc[-lookback] and dif.iloc[-lookback - 1] <= dea.iloc[-lookback - 1]:
                    score += 25
                    break

        # MACD 死叉
        for lookback in range(1, 6):
            if len(dif) > lookback + 1:
                if dif.iloc[-lookback] < dea.iloc[-lookback] and dif.iloc[-lookback - 1] >= dea.iloc[-lookback - 1]:
                    score -= 25
                    break

        # 红柱放大
        if len(macd_hist) >= 3:
            if macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3] > 0:
                score += 15

        # DIF > 0（在零轴上方）
        if dif.iloc[-1] > 0:
            score += 10

        return np.clip(score, 0, 100)

    def _score_rsi(self, df: pd.DataFrame) -> float:
        """
        RSI 打分：
        - 30~50 区间（超卖回升）→ 高分
        - 50~70 区间（强势）→ 中等
        - > 80（超买）→ 减分
        - < 20（深度超卖）→ 观望
        """
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]

        if pd.isna(last_rsi):
            return 50.0

        # 区间打分
        if 30 <= last_rsi <= 50:
            return 80  # 超卖回升，最佳买点
        elif 50 < last_rsi <= 65:
            return 65  # 强势
        elif 65 < last_rsi <= 80:
            return 45  # 偏高
        elif last_rsi > 80:
            return 20  # 超买
        elif last_rsi < 20:
            return 40  # 深度超卖，观望
        else:
            return 50

    def _score_trend(self, df: pd.DataFrame) -> float:
        """
        趋势打分：
        - 20 日线性回归斜率为正 → 加分
        - 收价在 20 日高低点的位置
        """
        close = df["close"].values
        if len(close) < 20:
            return 50.0

        # 线性回归斜率
        x = np.arange(20)
        y = close[-20:]
        slope = np.polyfit(x, y, 1)[0]
        norm_slope = slope / y.mean() * 100  # 归一化

        score = 50.0

        if norm_slope > 0.5:
            score += 25
        elif norm_slope > 0:
            score += 10
        elif norm_slope < -0.5:
            score -= 25
        elif norm_slope < 0:
            score -= 10

        # 收盘价在 20 日范围中的位置
        high_20 = max(close[-20:])
        low_20 = min(close[-20:])
        if high_20 != low_20:
            pos = (close[-1] - low_20) / (high_20 - low_20)
            if pos > 0.7:
                score += 10  # 接近新高
            elif pos < 0.3:
                score -= 10  # 接近新低

        return np.clip(score, 0, 100)

    # ──────────────────────────────────
    # 信号判定 + 仓位分配
    # ──────────────────────────────────

    def _assign_signals(self, results: list[StockScore]) -> list[StockScore]:
        """根据综合得分判定信号"""
        for r in results:
            # 技术面不够格的直接 HOLD
            if r.tech_score < self.min_tech_score:
                r.signal = "HOLD"
                r.reason = f"技术面不足 ({r.tech_score:.0f})"
                continue

            # ML 不够格的也 HOLD
            if self.ml_predictor is not None and r.ml_prob < self.ml_confidence:
                r.signal = "HOLD"
                r.reason = f"ML不足 ({r.ml_prob:.2f})"
                continue

            # 通过！
            r.signal = "BUY"
            parts = []
            if r.ma_score >= 70:
                parts.append("均线✓")
            if r.vol_score >= 70:
                parts.append("放量✓")
            if r.macd_score >= 70:
                parts.append("MACD✓")
            if r.ml_prob > 0:
                parts.append(f"ML:{r.ml_prob:.2f}")
            r.reason = " ".join(parts) if parts else "综合达标"

        return results

    def _allocate_positions(self, results: list[StockScore]) -> list[StockScore]:
        """
        按信号强度分配仓位

        逻辑：
        1. 只给 BUY 信号的股票分配仓位
        2. 以 final_score 为权重，按比例分配
        3. 单股不超过 max_position_pct
        """
        buy_list = [r for r in results if r.signal == "BUY"]
        if not buy_list:
            return results

        total_score = sum(r.final_score for r in buy_list)
        if total_score <= 0:
            return results

        for r in buy_list:
            raw_pct = (r.final_score / total_score) * 100
            r.position_pct = min(raw_pct, self.max_position_pct)

        # 归一化：确保总仓位不超过 60%（总仓位上限）
        total_pct = sum(r.position_pct for r in buy_list)
        if total_pct > 60:
            scale = 60 / total_pct
            for r in buy_list:
                r.position_pct *= scale

        return results

    # ──────────────────────────────────
    # 输出
    # ──────────────────────────────────

    @staticmethod
    def print_results(results: list[StockScore]):
        """格式化打印选股结果"""
        if not results:
            print("无结果")
            return

        print(f"\n{'='*80}")
        print(f"  📊 选股结果  {datetime.now():%Y-%m-%d %H:%M}")
        print(f"{'='*80}")
        print(f"{'排名':>4} {'代码':>8} {'名称':<8} {'价格':>8} {'技术分':>6} "
              f"{'ML概率':>6} {'综合分':>6} {'信号':>6} {'仓位%':>6} {'原因'}")
        print(f"{'-'*80}")

        for i, r in enumerate(results):
            icon = "🟢" if r.signal == "BUY" else "⬜"
            ml_str = f"{r.ml_prob:.2f}" if r.ml_prob > 0 else "  -"
            print(f"{i+1:>4} {r.symbol:>8} {r.name:<8} {r.price:>8.2f} "
                  f"{r.tech_score:>6.1f} {ml_str:>6} {r.final_score:>6.1f} "
                  f"{icon}{r.signal:>4} {r.position_pct:>5.1f}% {r.reason}")

        buy_list = [r for r in results if r.signal == "BUY"]
        if buy_list:
            total_pct = sum(r.position_pct for r in buy_list)
            print(f"\n  💡 建议买入: {len(buy_list)} 只 | 总仓位: {total_pct:.1f}%")
        else:
            print(f"\n  💤 今日无买入信号")

        print(f"{'='*80}")
