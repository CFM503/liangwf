"""
配置管理 — 读取 YAML，自动解密敏感字段
=======================================
所有参数集中管理，改参数不用改代码。
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field

from utils.crypto import maybe_decrypt

_CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"


@dataclass
class StrategyParams:
    """策略参数"""
    fast_period: int = 5
    slow_period: int = 20
    vol_period: int = 20
    vol_mult: float = 1.5
    stop_loss: float = 0.08
    take_profit: float = 0.15
    trailing_pct: float = 0.05


@dataclass
class RiskParams:
    """风控参数"""
    max_single_pct: float = 0.20
    max_total_pct: float = 0.60
    max_daily_loss_pct: float = 0.03
    limit_up_threshold: float = 0.095
    limit_down_threshold: float = -0.095
    kill_switch_file: str = ".kill_switch"


@dataclass
class MLParams:
    """机器学习参数"""
    enabled: bool = False
    model_type: str = "lightgbm"       # lightgbm / xgboost / random_forest / gradient_boosting
    lookback_days: int = 60
    retrain_days: int = 30
    forward_days: int = 3              # 预测未来 N 天
    threshold: float = 0.02            # 标签阈值（涨幅 > 2% 为正例）
    confidence_threshold: float = 0.6  # ML 置信度阈值
    n_estimators: int = 200
    max_depth: int = 5


@dataclass
class EmailParams:
    """邮件通知参数"""
    enabled: bool = False
    smtp_server: str = "smtp.qq.com"
    smtp_port: int = 465
    sender: str = ""
    password: str = ""
    receiver: str = ""


@dataclass
class BrokerParams:
    """券商参数"""
    mode: str = "simulator"
    api_key: str = ""
    api_secret: str = ""
    account_id: str = ""


@dataclass
class Config:
    """总配置"""
    stocks: list[str] = field(default_factory=lambda: ["600519", "300750", "601318"])
    stock_names: dict = field(default_factory=lambda: {
        "600519": "贵州茅台", "300750": "宁德时代", "601318": "中国平安",
        "000858": "五粮液", "600036": "招商银行",
    })
    initial_cash: float = 1_000_000
    run_time: str = "15:10"
    data_start: str = "20180101"
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    ml: MLParams = field(default_factory=MLParams)
    email: EmailParams = field(default_factory=EmailParams)
    broker: BrokerParams = field(default_factory=BrokerParams)
    log_dir: str = "logs"
    db_path: str = "xlt.db"


def load_config(path: str | Path | None = None) -> Config:
    """加载 YAML 配置文件"""
    path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not path.exists():
        print(f"[config] 未找到 {path}，使用默认参数")
        return Config()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    cfg.stocks = raw.get("stocks", cfg.stocks)
    cfg.stock_names = raw.get("stock_names", cfg.stock_names)
    cfg.initial_cash = raw.get("initial_cash", cfg.initial_cash)
    cfg.run_time = raw.get("run_time", cfg.run_time)
    cfg.data_start = raw.get("data_start", cfg.data_start)
    cfg.log_dir = raw.get("log_dir", cfg.log_dir)
    cfg.db_path = raw.get("db_path", cfg.db_path)

    for section, cls in [
        ("strategy", StrategyParams),
        ("risk", RiskParams),
        ("ml", MLParams),
    ]:
        if section in raw:
            setattr(cfg, section, cls(**{k: v for k, v in raw[section].items()
                                         if k in cls.__dataclass_fields__}))

    for section_name, cls in [("email", EmailParams), ("broker", BrokerParams)]:
        if section_name in raw:
            decrypted = {k: maybe_decrypt(v) if isinstance(v, str) else v
                         for k, v in raw[section_name].items()}
            setattr(cfg, section_name, cls(**{k: v for k, v in decrypted.items()
                                               if k in cls.__dataclass_fields__}))

    return cfg
