"""
时序滚动交叉验证框架 (Walk-Forward Validation Engine)
======================================================
严禁随机 K-Fold 交叉验证（避免未来数据泄漏与时间穿越）。
采用 Expanding Window 时序递增滚动切分：
- Fold 1: 训练 2018~2021 | 样本外测试 2022 (单边下行熊市)
- Fold 2: 训练 2018~2022 | 样本外测试 2023 (存量博弈震荡市)
- Fold 3: 训练 2018~2023 | 样本外测试 2024 (结构反弹政策市)

多模型横向全维度对比：
1. Baseline (纯双均线规则 + 量比)
2. Dual MA + LightGBM 过滤
3. Dual MA + XGBoost 过滤
4. 综合多维度指标：准确率、AUC、胜率、年化收益、最大回撤、夏普比率
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from data.fetcher import fetch_stock, STOCK_NAMES, DEFAULT_POOL
from ml_model.features import compute_features, FEATURE_COLS, make_label
from ml_model.predictor import MLPredictor, _create_model
from backtest.engine import BacktestEngine, DualMABTStrategy, ASharePandasData
import backtrader as bt
from utils.logger import get_logger

log = get_logger("xlt.validator")


class WalkForwardValidator:
    """
    时序滚动交叉验证器
    """

    def __init__(
        self,
        stock_codes: Optional[List[str]] = None,
        forward_days: int = 5,
        threshold: float = 0.03,
        ml_confidence: float = 0.55,
        initial_cash: float = 1_000_000,
    ):
        self.stock_codes = stock_codes or ["600519", "300750", "601318", "000858", "600036"]
        self.forward_days = forward_days
        self.threshold = threshold
        self.ml_confidence = ml_confidence
        self.initial_cash = initial_cash

    def _build_dataset(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """构建指定时间区间的特征与标签数据集"""
        all_features = []
        stock_dfs = {}

        for code in self.stock_codes:
            df = fetch_stock(code, start_date=start_date, end_date=end_date, use_cache=True)
            if df.empty or len(df) < 50:
                continue

            df_feat = compute_features(df)
            label = make_label(df, forward_days=self.forward_days, threshold=self.threshold)
            df_feat["label"] = label
            df_feat["code"] = code

            stock_dfs[code] = df_feat
            valid_rows = df_feat[FEATURE_COLS + ["label", "code"]].dropna()
            all_features.append(valid_rows)

        if not all_features:
            return pd.DataFrame(), {}

        dataset = pd.concat(all_features).sort_index()
        return dataset, stock_dfs

    def _run_backtest_with_prob(
        self,
        stock_dfs: Dict[str, pd.DataFrame],
        models_dict: Dict[str, MLPredictor],
        test_start: str,
        test_end: str,
        strategy_type: str = "baseline",
    ) -> dict:
        """
        在测试集区间运行带有真实约束的 Backtrader 回测。
        strategy_type: 'baseline', 'lightgbm', 'xgboost'
        """
        cerebro = bt.Cerebro()

        # 加载各股票在测试区间的行情与 ML 预测值
        loaded = 0
        for code, full_df in stock_dfs.items():
            # 截取测试区间 (转为 datetime 比较)
            dt_start = pd.to_datetime(test_start)
            dt_end = pd.to_datetime(test_end)
            mask = (full_df.index >= dt_start) & (full_df.index <= dt_end)
            test_df = full_df.loc[mask].copy()

            if test_df.empty or len(test_df) < 20:
                continue

            # 填充 ML 概率
            if strategy_type in models_dict and models_dict[strategy_type].model is not None:
                predictor = models_dict[strategy_type]
                X_test = test_df[FEATURE_COLS].fillna(0)
                probs = predictor.model.predict_proba(X_test)[:, 1]
                test_df["ml_prob"] = probs
            else:
                test_df["ml_prob"] = 0.5

            data = ASharePandasData(
                dataname=test_df,
                name=code,
                datetime=None,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                openinterest=-1,
                amount="amount" if "amount" in test_df.columns else -1,
                turnover="turnover" if "turnover" in test_df.columns else -1,
                limit_up="limit_up" if "limit_up" in test_df.columns else -1,
                limit_down="limit_down" if "limit_down" in test_df.columns else -1,
                is_limit_up="is_limit_up" if "is_limit_up" in test_df.columns else -1,
                is_limit_down="is_limit_down" if "is_limit_down" in test_df.columns else -1,
                is_suspended="is_suspended" if "is_suspended" in test_df.columns else -1,
                ml_prob="ml_prob",
            )
            cerebro.adddata(data)
            loaded += 1

        if loaded == 0:
            return {"error": "无有效测试数据"}

        # 策略参数配置
        use_ml = (strategy_type != "baseline")
        cerebro.addstrategy(
            DualMABTStrategy,
            fast_period=5,
            slow_period=20,
            vol_mult=1.5,
            use_ml=use_ml,
            ml_confidence=self.ml_confidence,
        )

        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=0.00025)
        cerebro.broker.set_slippage_perc(0.001)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.03)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        results = cerebro.run()
        strat = results[0]

        trades = strat.trade_log
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]

        sharpe_data = strat.analyzers.sharpe.get_analysis()
        dd_data = strat.analyzers.drawdown.get_analysis()
        ret_data = strat.analyzers.returns.get_analysis()

        annual_ret = ret_data.get("rnorm100", 0)
        max_dd = dd_data.get("max", {}).get("drawdown", 0)
        sharpe = sharpe_data.get("sharperatio", 0) or 0
        final_val = cerebro.broker.getvalue()

        win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
        return {
            "trades_count": len(trades),
            "win_rate": win_rate,
            "annual_return": annual_ret,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "final_value": final_val,
        }

    def run_walk_forward(self, custom_folds: Optional[List[dict]] = None) -> dict:
        """
        执行完整的 Walk-Forward 时序滚动验证。
        """
        folds = custom_folds or [
            {
                "name": "Fold 1 (2022 熊市测试)",
                "train_start": "20180101",
                "train_end": "20211231",
                "test_start": "20220101",
                "test_end": "20221231",
            },
            {
                "name": "Fold 2 (2023 震荡市测试)",
                "train_start": "20180101",
                "train_end": "20221231",
                "test_start": "20230101",
                "test_end": "20231231",
            },
            {
                "name": "Fold 3 (2024 反弹市测试)",
                "train_start": "20180101",
                "train_end": "20231231",
                "test_start": "20240101",
                "test_end": "20241231",
            },
        ]

        fold_results = []

        for fold in folds:
            log.info(f"========== 正在执行: {fold['name']} ==========")
            # 1. 构建训练与测试数据集
            train_df, _ = self._build_dataset(fold["train_start"], fold["train_end"])
            test_df, stock_dfs = self._build_dataset(fold["train_start"], fold["test_end"])

            # 提取测试区间数据用于分类评估
            dt_test_start = pd.to_datetime(fold["test_start"])
            dt_test_end = pd.to_datetime(fold["test_end"])
            test_eval_df = test_df[(test_df.index >= dt_test_start) & (test_df.index <= dt_test_end)]

            X_train = train_df[FEATURE_COLS]
            y_train = train_df["label"].astype(int)

            X_test = test_eval_df[FEATURE_COLS]
            y_test = test_eval_df["label"].astype(int)

            # 2. 训练 LightGBM
            lgb_pred = MLPredictor(model_type="lightgbm", n_estimators=150, max_depth=5, learning_rate=0.03)
            lgb_pred.model = _create_model("lightgbm", 150, 5, 0.03)
            lgb_pred.model.fit(X_train, y_train)

            # 3. 训练 XGBoost
            xgb_pred = MLPredictor(model_type="xgboost", n_estimators=150, max_depth=4, learning_rate=0.03)
            xgb_pred.model = _create_model("xgboost", 150, 4, 0.03)
            xgb_pred.model.fit(X_train, y_train)

            models_dict = {
                "lightgbm": lgb_pred,
                "xgboost": xgb_pred,
            }

            # 4. 计算测试集分类指标 (OOS)
            # LightGBM
            lgb_prob = lgb_pred.model.predict_proba(X_test)[:, 1]
            lgb_pred_y = (lgb_prob >= 0.5).astype(int)
            lgb_acc = accuracy_score(y_test, lgb_pred_y)
            lgb_auc = roc_auc_score(y_test, lgb_prob) if len(np.unique(y_test)) > 1 else 0.5
            lgb_prec = precision_score(y_test, lgb_pred_y, zero_division=0)
            lgb_rec = recall_score(y_test, lgb_pred_y, zero_division=0)

            # XGBoost
            xgb_prob = xgb_pred.model.predict_proba(X_test)[:, 1]
            xgb_pred_y = (xgb_prob >= 0.5).astype(int)
            xgb_acc = accuracy_score(y_test, xgb_pred_y)
            xgb_auc = roc_auc_score(y_test, xgb_prob) if len(np.unique(y_test)) > 1 else 0.5
            xgb_prec = precision_score(y_test, xgb_pred_y, zero_division=0)
            xgb_rec = recall_score(y_test, xgb_pred_y, zero_division=0)

            # 5. 回测评估 (Baseline vs LightGBM vs XGBoost)
            bt_base = self._run_backtest_with_prob(stock_dfs, models_dict, fold["test_start"], fold["test_end"], "baseline")
            bt_lgb = self._run_backtest_with_prob(stock_dfs, models_dict, fold["test_start"], fold["test_end"], "lightgbm")
            bt_xgb = self._run_backtest_with_prob(stock_dfs, models_dict, fold["test_start"], fold["test_end"], "xgboost")

            fold_summary = {
                "fold_name": fold["name"],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "pos_ratio_test": f"{y_test.mean()*100:.1f}%",
                "classification": {
                    "lightgbm": {"acc": lgb_acc, "auc": lgb_auc, "precision": lgb_prec, "recall": lgb_rec},
                    "xgboost": {"acc": xgb_acc, "auc": xgb_auc, "precision": xgb_prec, "recall": xgb_rec},
                },
                "backtest": {
                    "baseline": bt_base,
                    "lightgbm": bt_lgb,
                    "xgboost": bt_xgb,
                }
            }
            fold_results.append(fold_summary)

        return {"folds": fold_results}

    def print_report(self, results: dict):
        """打印时序滚动验证报告"""
        print("\n" + "=" * 80)
        print("          📊 时序滚动交叉验证 (Walk-Forward Validation) 结果报告")
        print("=" * 80)

        for f in results.get("folds", []):
            print(f"\n▶ 【{f['fold_name']}】 (训练集: {f['train_samples']} 条, 样本外测试集: {f['test_samples']} 条, 正样本率: {f['pos_ratio_test']})")
            print("-" * 80)
            print("  1. 样本外分类能力 (OOS Metrics):")
            print(f"     • LightGBM : 准确率 {f['classification']['lightgbm']['acc']*100:.2f}% | AUC {f['classification']['lightgbm']['auc']:.4f} | 精准率 {f['classification']['lightgbm']['precision']*100:.2f}% | 召回率 {f['classification']['lightgbm']['recall']*100:.2f}%")
            print(f"     • XGBoost  : 准确率 {f['classification']['xgboost']['acc']*100:.2f}% | AUC {f['classification']['xgboost']['auc']:.4f} | 精准率 {f['classification']['xgboost']['precision']*100:.2f}% | 召回率 {f['classification']['xgboost']['recall']*100:.2f}%")
            print("\n  2. 样本外真实交易绩效对比 (含 T+1 / 涨跌停约束):")
            print(f"     {'策略 / 模型':<20s} {'交易次数':<10s} {'胜率':<10s} {'年化收益':<12s} {'最大回撤':<12s} {'夏普比率':<10s}")
            print("     " + "-" * 72)
            
            for strat_key, strat_name in [("baseline", "Baseline (纯双均线)"), ("lightgbm", "Dual MA + LightGBM"), ("xgboost", "Dual MA + XGBoost")]:
                bt_m = f["backtest"].get(strat_key, {})
                t_cnt = str(bt_m.get("trades_count", 0))
                w_rate = f"{bt_m.get('win_rate', 0):.1f}%"
                ann_ret = f"{bt_m.get('annual_return', 0):+.2f}%"
                m_dd = f"{bt_m.get('max_drawdown', 0):.2f}%"
                shp = f"{bt_m.get('sharpe_ratio', 0):.2f}"
                print(f"     {strat_name:<20s} {t_cnt:<10s} {w_rate:<10s} {ann_ret:<12s} {m_dd:<12s} {shp:<10s}")

        print("=" * 80)


if __name__ == "__main__":
    validator = WalkForwardValidator()
    results = validator.run_walk_forward()
    validator.print_report(results)
