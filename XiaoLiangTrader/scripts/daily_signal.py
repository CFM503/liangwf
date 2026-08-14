"""
每日买卖点预测 Pipeline — 命令行一键输出交易信号
===================================================
运行命令:
    python XiaoLiangTrader/scripts/daily_signal.py
    python XiaoLiangTrader/scripts/daily_signal.py --pool 600519 300750 002594
    python XiaoLiangTrader/scripts/daily_signal.py --confidence 0.55
    python XiaoLiangTrader/scripts/daily_signal.py --date 20241231

功能特性:
1. 明确输出："股票代码 + 名称 + 建议动作 (BUY/SELL) + 价格 + 止损止盈位 + ML置信度 + 触发原因"
2. 技术面 (Dual MA 5/20 + 放量) + LightGBM 置信度双重过滤
3. 严格排除停牌、ST、触及涨跌停无法成交的标的
4. 结果同时保存为 JSON 供程序化消费与审计
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Add XiaoLiangTrader root to sys.path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np

from data.fetcher import fetch_stock, STOCK_NAMES, DEFAULT_POOL, fetch_all_a_snapshot
from strategy.signals import Action, Signal
from strategy.dual_ma import DualMAStrategy
from ml_model.predictor import MLPredictor
from ml_model.features import compute_features, FEATURE_COLS
from utils.logger import get_logger

log = get_logger("xlt.signal")

RESULTS_DIR = _root / "ml_model" / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class DailySignalPipeline:
    """
    每日买卖点信号生成流水线
    """

    def __init__(
        self,
        ml_confidence: float = 0.55,
        fast_period: int = 5,
        slow_period: int = 20,
        vol_mult: float = 1.5,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.15,
        model_path: Optional[str] = None,
    ):
        self.ml_confidence = ml_confidence
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_mult = vol_mult
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # 基础双均线策略
        self.strategy = DualMAStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
            vol_mult=vol_mult,
            stop_loss=stop_loss_pct,
            take_profit=take_profit_pct,
        )

        # 加载或初始化 LightGBM 预测器
        self.predictor = MLPredictor(
            model_type="lightgbm",
            forward_days=5,
            threshold=0.03,
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
        )

        # 尝试加载最新已保存的模型
        models = self.predictor.list_saved_models()
        if models and model_path is None:
            try:
                self.predictor.load(models[-1])
            except Exception as e:
                log.warning(f"加载已有模型失败: {e}，将在首次使用时就地训练")
        elif model_path:
            self.predictor.load(model_path)

    def _ensure_model_trained(self, train_codes: List[str], train_end_date: str):
        """若无可用模型，使用历史数据自动拟合"""
        if self.predictor.model is not None:
            return

        log.info("[Pipeline] 正在为 LightGBM 模型进行全样本池拟合...")
        train_start = (datetime.strptime(train_end_date, "%Y%m%d") - timedelta(days=365 * 4)).strftime("%Y%m%d")
        
        all_feats = []
        for code in train_codes[:30]:  # 使用前30只代表股快速训练
            df = fetch_stock(code, start_date=train_start, end_date=train_end_date, use_cache=True)
            if df.empty or len(df) < 100:
                continue
            df_feat = compute_features(df)
            from ml_model.features import make_label
            df_feat["label"] = make_label(df, forward_days=5, threshold=0.03)
            all_feats.append(df_feat[FEATURE_COLS + ["label"]].dropna())

        if all_feats:
            train_df = pd.concat(all_feats)
            X = train_df[FEATURE_COLS]
            y = train_df["label"].astype(int)
            from ml_model.predictor import _create_model
            self.predictor.model = _create_model("lightgbm", 150, 5, 0.03)
            self.predictor.model.fit(X, y)
            self.predictor.train_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.predictor.save()
            log.info("[Pipeline] LightGBM 模型训练并保存完成")

    def scan_signals(
        self,
        stock_codes: List[str],
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """
        对指定股票池扫描买卖信号。

        Args:
            stock_codes: 待扫描股票代码列表
            as_of_date: 计算截止日期 (YYYYMMDD，默认最新)

        Returns:
            包含信号列表与扫描统计的 dict
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime("%Y%m%d")

        # 保证模型已训练
        self._ensure_model_trained(stock_codes, as_of_date)

        start_date = (datetime.strptime(as_of_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        signals = []
        skipped_count = 0
        total_scanned = 0
        actual_date = None

        for code in stock_codes:
            name = STOCK_NAMES.get(code, code)
            df = fetch_stock(code, start_date=start_date, end_date=as_of_date, use_cache=True)
            if df.empty or len(df) < self.slow_period + 5:
                skipped_count += 1
                continue

            total_scanned += 1
            latest_bar = df.iloc[-1]
            bar_date = df.index[-1].strftime("%Y-%m-%d")
            actual_date = bar_date
            price = float(latest_bar["close"])

            # 1. 停牌过滤
            is_suspended = bool(latest_bar.get("is_suspended", False)) or (latest_bar["volume"] <= 0)
            if is_suspended:
                skipped_count += 1
                continue

            # 2. ST 股票过滤
            is_st = bool(latest_bar.get("is_st", False)) or ("ST" in name)
            if is_st:
                skipped_count += 1
                continue

            # 3. 涨跌停判定
            is_limit_up = bool(latest_bar.get("is_limit_up", False))
            is_limit_down = bool(latest_bar.get("is_limit_down", False))

            # 4. 计算均线与指标
            df_ind = self.strategy.compute_indicators(df)
            latest_ind = df_ind.iloc[-1]
            golden_cross = bool(latest_ind["golden_cross"])
            death_cross = bool(latest_ind["death_cross"])
            vol_ok = bool(latest_ind["vol_ok"])
            vol_ratio = float(latest_ind["volume"] / latest_ind["vol_ma"]) if latest_ind["vol_ma"] > 0 else 1.0

            # 5. ML 置信度预测
            ml_prob = 0.5
            if self.predictor.model is not None:
                try:
                    df_feat = compute_features(df)
                    X_latest = df_feat[FEATURE_COLS].iloc[[-1]].fillna(0)
                    ml_prob = float(self.predictor.model.predict_proba(X_latest)[0][1])
                except Exception:
                    ml_prob = 0.5

            # ── 信号判定 ──
            # 买入条件：金叉 + 放量 + ML置信度达标 + 未触及涨停
            if golden_cross and vol_ok and not is_limit_up:
                if ml_prob >= self.ml_confidence:
                    stop_loss_price = round(price * (1.0 - self.stop_loss_pct), 2)
                    take_profit_price = round(price * (1.0 + self.take_profit_pct), 2)
                    signals.append({
                        "symbol": code,
                        "name": name,
                        "action": Action.BUY.value,
                        "action_label": "🟢 建议买入",
                        "price": round(price, 2),
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "ml_confidence": round(ml_prob, 3),
                        "vol_ratio": round(vol_ratio, 2),
                        "reason": f"MA{self.fast_period}/20金叉 + 放量{vol_ratio:.1f}x + LightGBM置信度{ml_prob:.2f}",
                        "date": bar_date,
                    })

            # 卖出条件：死叉 + 未封死跌停
            elif death_cross and not is_limit_down:
                signals.append({
                    "symbol": code,
                    "name": name,
                    "action": Action.SELL.value,
                    "action_label": "🔴 建议卖出",
                    "price": round(price, 2),
                    "stop_loss_price": 0.0,
                    "take_profit_price": 0.0,
                    "ml_confidence": round(ml_prob, 3),
                    "vol_ratio": round(vol_ratio, 2),
                    "reason": f"MA{self.fast_period}/20死叉平仓",
                    "date": bar_date,
                })

        # 按置信度降序排序
        signals.sort(key=lambda s: (s["action"] != Action.BUY.value, -s["ml_confidence"]))

        result_payload = {
            "date": actual_date or as_of_date,
            "total_pool_size": len(stock_codes),
            "scanned_count": total_scanned,
            "signals_triggered_count": len(signals),
            "buy_count": sum(1 for s in signals if s["action"] == Action.BUY.value),
            "sell_count": sum(1 for s in signals if s["action"] == Action.SELL.value),
            "no_signal_count": total_scanned - len(signals),
            "ml_confidence_threshold": self.ml_confidence,
            "signals": signals,
            "disclaimer": "⚠️ 声明：本信号仅供量化模型学习与学术研究参考，不构成任何实际投资建议。股市有风险，入市需谨慎。",
        }

        return result_payload

    def print_signal_report(self, payload: Dict):
        """终端结构化打印买卖点信号看板"""
        date_str = payload.get("date", datetime.now().strftime("%Y-%m-%d"))
        signals = payload.get("signals", [])
        total_scanned = payload.get("scanned_count", 0)
        buy_cnt = payload.get("buy_count", 0)
        sell_cnt = payload.get("sell_count", 0)
        no_sig_cnt = payload.get("no_signal_count", 0)
        conf_thr = payload.get("ml_confidence_threshold", 0.55)

        print("\n" + "=" * 90)
        print(f"       🎯 XiaoLiangTrader 每日买卖点预测看板 ({date_str})")
        print(f"       模型引擎: LightGBM (置信度门槛 ≥ {conf_thr:.2f}) | 规则引擎: MA5/20 + 放量")
        print("=" * 90)

        if not signals:
            print("\n  💡 今日扫描完成，无明确触发买卖点的标的（全市场处于震荡或持仓观望状态）。")
        else:
            print(f"\n{'代码':<8s} {'名称':<8s} {'建议动作':<12s} {'最新价':<9s} {'建议止损':<9s} {'建议止盈':<9s} {'ML置信度':<9s} {'触发原因'}")
            print("-" * 90)
            for s in signals:
                sym = s["symbol"]
                name = s["name"]
                action = s["action_label"]
                p = f"{s['price']:.2f}"
                sl = f"{s['stop_loss_price']:.2f}" if s['stop_loss_price'] > 0 else "-"
                tp = f"{s['take_profit_price']:.2f}" if s['take_profit_price'] > 0 else "-"
                conf = f"{s['ml_confidence']:.2f}"
                reason = s["reason"]
                print(f"{sym:<8s} {name:<8s} {action:<12s} {p:<9s} {sl:<9s} {tp:<9s} {conf:<9s} {reason}")

        print("-" * 90)
        print(f"📊 扫描统计: 已扫描 {total_scanned} 只股票 | 触发买入: {buy_cnt} 只 | 触发卖出: {sell_cnt} 只 | 观望/未触发: {no_sig_cnt} 只")
        print("=" * 90)
        print(f"{payload.get('disclaimer')}")
        print("=" * 90 + "\n")


    def format_notification_text(self, payload: Dict) -> str:
        """生成用于邮件和推送的纯文本报告"""
        date_str = payload.get("date", datetime.now().strftime("%Y-%m-%d"))
        signals = payload.get("signals", [])
        total_scanned = payload.get("scanned_count", 0)
        buy_cnt = payload.get("buy_count", 0)
        sell_cnt = payload.get("sell_count", 0)
        no_sig_cnt = payload.get("no_signal_count", 0)

        lines = [
            f"🎯 XiaoLiangTrader 每日买卖点预测看板 ({date_str})",
            "=" * 60,
            f"📊 扫描概况: 扫描 {total_scanned} 只 | 买入 {buy_cnt} 只 | 卖出 {sell_cnt} 只 | 观望 {no_sig_cnt} 只",
            "-" * 60,
        ]

        if not signals:
            lines.append("今日无触发买卖点的标的（全市场处于震荡或持仓观望状态）。")
        else:
            lines.append("【触发信号明细】")
            for s in signals:
                action = s["action_label"]
                sym = s["symbol"]
                name = s["name"]
                p = s["price"]
                sl = f"{s['stop_loss_price']:.2f}" if s['stop_loss_price'] > 0 else "-"
                tp = f"{s['take_profit_price']:.2f}" if s['take_profit_price'] > 0 else "-"
                conf = s["ml_confidence"]
                reason = s["reason"]
                lines.append(f"• {action} | {sym} {name} | 现价:{p:.2f} | 止损:{sl} | 止盈:{tp} | ML:{conf:.2f} | {reason}")

        lines.extend([
            "-" * 60,
            f"{payload.get('disclaimer', '')}",
            "=" * 60,
        ])
        return "\n".join(lines)

    def send_notification(self, payload: Dict, config_path: Optional[str] = None) -> bool:
        """调用 Notifier 发送邮件通知"""
        from config.settings import load_config
        from bot.notifier import Notifier

        cfg = load_config(config_path)
        notifier = Notifier(
            enabled=cfg.email.enabled,
            smtp_server=cfg.email.smtp_server,
            smtp_port=cfg.email.smtp_port,
            sender=cfg.email.sender,
            password=cfg.email.password,
            receiver=cfg.email.receiver,
        )

        date_str = payload.get("date", datetime.now().strftime("%Y-%m-%d"))
        text_body = self.format_notification_text(payload)
        buy_cnt = payload.get("buy_count", 0)
        sell_cnt = payload.get("sell_count", 0)

        subject = f"每日信号看板 ({date_str}) [买入:{buy_cnt} / 卖出:{sell_cnt}]"
        
        if not notifier.enabled or not notifier.sender:
            log.info(f"[通知] 邮件通知未配置或未启用 (可在 config/config.yaml 中开启)。")
            log.info(f"[通知内容预览]:\n{text_body}")
            return False

        sent = notifier.notify_report(text_body)
        if sent:
            log.info(f"[通知] 邮件通知已成功发送至 {notifier.receiver}")
        return sent


def main():
    parser = argparse.ArgumentParser(description="XiaoLiangTrader 每日买卖点预测 Pipeline")
    parser.add_argument("--pool", nargs="+", help="指定股票代码列表（默认 60 只核心龙头池）")
    parser.add_argument("--confidence", type=float, default=0.55, help="LightGBM 置信度阈值 (默认 0.55)")
    parser.add_argument("--date", type=str, default=None, help="指定计算日期 (YYYYMMDD，默认最新行情)")
    parser.add_argument("--all", action="store_true", help="全市场扫描模式")
    parser.add_argument("--output", type=str, default=None, help="导出 JSON 结果路径")
    parser.add_argument("--notify", action="store_true", help="触发信号时发送邮件通知")
    parser.add_argument("--schedule", action="store_true", help="启动收盘后定时自动扫描守护进程")
    parser.add_argument("--time", type=str, default="15:10", help="定时运行时间 (HH:MM，默认 15:10)")

    args = parser.parse_args()

    # 确定股票池
    if args.all:
        print("[Pipeline] 正在拉取全 A 股实时快照...")
        spot = fetch_all_a_snapshot()
        if not spot.empty and "code" in spot.columns:
            target_codes = spot["code"].tolist()
        else:
            target_codes = DEFAULT_POOL
    elif args.pool:
        target_codes = args.pool
    else:
        target_codes = DEFAULT_POOL

    pipeline = DailySignalPipeline(ml_confidence=args.confidence)

    def run_once():
        payload = pipeline.scan_signals(target_codes, as_of_date=args.date)
        pipeline.print_signal_report(payload)

        # 导出 JSON
        output_path = Path(args.output) if args.output else (RESULTS_DIR / "daily_signals_latest.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        log.info(f"[Pipeline] 信号结果已导出至: {output_path}")

        # 邮件通知
        if args.notify:
            pipeline.send_notification(payload)

    if args.schedule:
        try:
            import schedule
            import time
        except ImportError:
            print("请先安装: pip install schedule")
            sys.exit(1)

        print("=" * 60)
        print(f"⏰ XiaoLiangTrader 收盘后定时信号守护已启动")
        print(f"   每日定点运行时间: {args.time} (A股收盘后)")
        print(f"   扫描标的规模: {len(target_codes)} 只")
        print(f"   邮件通知: {'开启' if args.notify else '关闭'}")
        print("   按 Ctrl+C 退出守护进程")
        print("=" * 60)

        schedule.every().day.at(args.time).do(run_once)

        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        run_once()


if __name__ == "__main__":
    main()
