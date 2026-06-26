"""
统一 ML 预测器 — 支持 LightGBM / XGBoost / sklearn
====================================================
2008-2010 那会儿还没有 LightGBM，用的是 sklearn 的
RandomForest 和 GradientBoosting。后来 XGBoost 出了，
宿舍里第一时间试了。三种模型各有优劣，这里统一接口。

模型选择（config.yaml 里 ml.model_type）:
  - "lightgbm"          速度最快，精度不错（推荐）
  - "xgboost"           精度略高，速度稍慢
  - "random_forest"     最稳定，不容易过拟合
  - "gradient_boosting" sklearn 自带的 GBDT，兼容性好

内存占用：全部 < 100MB，16GB 内存绰绰有余。
训练时间：7 年日线数据（~1700 条），笔记本上 < 5 秒。
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from .features import compute_features, FEATURE_COLS, make_label
from utils.logger import get_logger

log = get_logger("xlt.ml")

# 模型存储目录
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


def _create_model(model_type: str, n_estimators: int, max_depth: int, learning_rate: float):
    """
    根据类型创建模型实例。
    这里做了兼容处理：没装 lightgbm/xgboost 会自动降级到 sklearn。
    """
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            log.warning("[ML] lightgbm 未安装，降级到 sklearn GradientBoosting")

    elif model_type == "xgboost":
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        except ImportError:
            log.warning("[ML] xgboost 未安装，降级到 sklearn GradientBoosting")

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )

    # 默认 / 降级：sklearn GradientBoosting
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )


class MLPredictor:
    """
    统一 ML 预测器

    用法:
        predictor = MLPredictor(model_type="lightgbm")
        metrics = predictor.train(df)      # 训练
        prob = predictor.predict_proba(df)  # 预测
        predictor.save()                    # 保存
        predictor.load("model_20240101.pkl")  # 加载
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        forward_days: int = 3,
        threshold: float = 0.02,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
    ):
        self.model_type = model_type
        self.forward_days = forward_days
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.train_date: str = ""

    def train(self, df: pd.DataFrame) -> dict:
        """
        训练模型。

        Args:
            df: 原始 OHLCV DataFrame（至少 200 行）

        Returns:
            训练指标 dict
        """
        log.info(f"[ML] 开始训练 ({self.model_type})...")

        # 1. 特征 + 标签
        df_feat = compute_features(df)
        label = make_label(df, self.forward_days, self.threshold)

        df_train = df_feat[FEATURE_COLS].copy()
        df_train["label"] = label
        df_train = df_train.dropna()

        if len(df_train) < 100:
            raise ValueError(f"训练数据不足: {len(df_train)} 条")

        X = df_train[FEATURE_COLS]
        y = df_train["label"]

        # 2. 按时间切分（前 80% 训练，后 20% 验证）
        # 注意：时间序列不能随机切分，否则会用未来数据训练
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 3. 创建并训练
        self.model = _create_model(
            self.model_type, self.n_estimators, self.max_depth, self.learning_rate
        )
        self.model.fit(X_train, y_train)

        # 4. 评估
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        self.train_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 特征重要性
        importance = {}
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            importance = dict(zip(FEATURE_COLS, imp))
            importance = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])

        metrics = {
            "model_type": self.model_type,
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "top_features": importance,
            "train_date": self.train_date,
        }

        log.info(f"[ML] 训练完成 | 训练 {train_acc:.2%} | 验证 {val_acc:.2%}")
        if importance:
            log.info(f"[ML] Top5: {list(importance.keys())[:5]}")

        return metrics

    def predict_proba(self, df: pd.DataFrame) -> float:
        """
        预测最新 K 线的"未来上涨概率"。

        Returns:
            0~1 的概率值
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        df_feat = compute_features(df)
        latest = df_feat[FEATURE_COLS].iloc[[-1]]

        if latest.isnull().any().any():
            log.warning("[ML] 最新数据含 NaN，返回 0.5")
            return 0.5

        prob = self.model.predict_proba(latest)[0][1]
        return float(prob)

    def save(self, filename: str | None = None):
        """保存模型"""
        if self.model is None:
            raise RuntimeError("无模型可保存")
        if filename is None:
            filename = f"ml_{self.model_type}_{datetime.now():%Y%m%d_%H%M}.pkl"
        path = MODEL_DIR / filename
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "model_type": self.model_type,
                "train_date": self.train_date,
                "params": {
                    "forward_days": self.forward_days,
                    "threshold": self.threshold,
                },
            }, f)
        log.info(f"[ML] 模型已保存: {path}")

    def load(self, filename: str):
        """加载模型"""
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.model_type = data.get("model_type", "unknown")
        self.train_date = data.get("train_date", "unknown")
        log.info(f"[ML] 已加载: {path} ({self.model_type}, 训练于 {self.train_date})")

    def list_saved_models(self) -> list[str]:
        """列出所有已保存的模型"""
        return sorted([f.name for f in MODEL_DIR.glob("ml_*.pkl")])
