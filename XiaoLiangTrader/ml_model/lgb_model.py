"""
LightGBM 轻量预测模型 — 宿舍笔记本也能跑
==========================================
模型很简单：用过去的技术指标预测未来 5 天上涨概率。
不用 GPU，几秒就能训练完。
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from .features import compute_features, FEATURE_COLS, make_label
from utils.logger import get_logger

log = get_logger("xlt.lgb")

# 模型缓存目录
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


class LightGBMModel:
    """
    LightGBM 分类器 — 预测"未来5天上涨概率"

    用法:
        model = LightGBMModel()
        model.train(df)                    # 训练
        prob = model.predict_proba(df)     # 预测概率
        model.save("my_model.pkl")         # 保存
        model.load("my_model.pkl")         # 加载
    """

    def __init__(
        self,
        forward_days: int = 5,
        threshold: float = 0.02,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
    ):
        self.forward_days = forward_days
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.train_date: str = ""

    def train(self, df: pd.DataFrame) -> dict:
        """
        训练 LightGBM 模型。

        Args:
            df: 原始 OHLCV DataFrame

        Returns:
            训练指标 dict（accuracy, feature_importance）
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("请安装 lightgbm: pip install lightgbm")

        log.info("[LGB] 开始训练...")

        # 1. 计算特征和标签
        df_feat = compute_features(df)
        label = make_label(df, self.forward_days, self.threshold)

        # 2. 合并并去除 NaN
        df_train = df_feat[FEATURE_COLS].copy()
        df_train["label"] = label
        df_train = df_train.dropna()

        if len(df_train) < 100:
            raise ValueError(f"训练数据不足: {len(df_train)} 条（需要至少 100 条）")

        X = df_train[FEATURE_COLS]
        y = df_train["label"]

        # 3. 按时间切分：前 80% 训练，后 20% 验证
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 4. 训练
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,  # 不打印训练日志
        )
        self.model.fit(X_train, y_train)

        # 5. 评估
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        self.train_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 特征重要性
        importance = dict(zip(FEATURE_COLS, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])

        metrics = {
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "top_features": importance,
            "train_date": self.train_date,
        }

        log.info(f"[LGB] 训练完成 | 训练集准确率 {train_acc:.2%} | 验证集 {val_acc:.2%}")
        log.info(f"[LGB] Top5 特征: {list(importance.keys())[:5]}")

        return metrics

    def predict_proba(self, df: pd.DataFrame) -> float:
        """
        预测最新一根 K 线的"上涨概率"。

        Args:
            df: 原始 OHLCV DataFrame（至少 60 行）

        Returns:
            0~1 之间的概率值
        """
        if self.model is None:
            raise RuntimeError("模型未训练/加载，请先调用 train() 或 load()")

        df_feat = compute_features(df)
        latest = df_feat[FEATURE_COLS].iloc[[-1]]

        # 检查是否有 NaN
        if latest.isnull().any().any():
            log.warning("[LGB] 最新数据含 NaN，返回 0.5")
            return 0.5

        prob = self.model.predict_proba(latest)[0][1]
        return float(prob)

    def save(self, filename: str | None = None):
        """保存模型到文件"""
        if self.model is None:
            raise RuntimeError("无模型可保存")
        if filename is None:
            filename = f"lgb_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        path = MODEL_DIR / filename
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "train_date": self.train_date,
                "params": {
                    "forward_days": self.forward_days,
                    "threshold": self.threshold,
                }
            }, f)
        log.info(f"[LGB] 模型已保存: {path}")

    def load(self, filename: str):
        """从文件加载模型"""
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.train_date = data.get("train_date", "unknown")
        log.info(f"[LGB] 模型已加载: {path} (训练于 {self.train_date})")

    def list_saved_models(self) -> list[str]:
        """列出所有已保存的模型文件"""
        return [f.name for f in MODEL_DIR.glob("lgb_*.pkl")]
