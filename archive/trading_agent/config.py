"""
配置管理 — 读取 YAML 配置，自动解密敏感字段
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
import yaml

from .crypto import decrypt_config_value

CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"


@dataclass
class StrategyConfig:
    fast_period: int = 5
    slow_period: int = 20
    vol_period: int = 20
    vol_mult: float = 1.5
    stop_loss: float = 0.08
    take_profit: float = 0.15
    trailing_pct: float = 0.05


@dataclass
class RiskConfig:
    max_single_pct: float = 0.20
    max_total_pct: float = 0.60
    max_daily_loss_pct: float = 0.03
    kill_switch_file: str = ".kill_switch"


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_server: str = "smtp.qq.com"
    smtp_port: int = 465
    sender: str = ""
    password: str = ""  # 加密存储
    receiver: str = ""


@dataclass
class BrokerConfig:
    mode: str = "simulator"  # simulator / live
    api_key: str = ""        # 加密存储
    api_secret: str = ""     # 加密存储
    account_id: str = ""


@dataclass
class TradingConfig:
    stocks: list[str] = field(default_factory=lambda: ["600519", "300750", "601318"])
    initial_cash: float = 1_000_000
    run_time: str = "15:10"  # 每日运行时间（收盘后）
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    log_dir: str = "logs"
    db_path: str = "trading_agent.db"


def load_config(path: str | Path | None = None) -> TradingConfig:
    """加载配置文件，自动解密 ENC: 字段"""
    path = Path(path) if path else DEFAULT_CONFIG

    if not path.exists():
        print(f"[config] 配置文件不存在: {path}，使用默认配置")
        return TradingConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 解密敏感字段
    for section in ["email", "broker"]:
        if section in raw:
            for key, val in raw[section].items():
                raw[section][key] = decrypt_config_value(val)

    # 构建 dataclass
    cfg = TradingConfig()
    cfg.stocks = raw.get("stocks", cfg.stocks)
    cfg.initial_cash = raw.get("initial_cash", cfg.initial_cash)
    cfg.run_time = raw.get("run_time", cfg.run_time)
    cfg.log_dir = raw.get("log_dir", cfg.log_dir)
    cfg.db_path = raw.get("db_path", cfg.db_path)

    if "strategy" in raw:
        cfg.strategy = StrategyConfig(**raw["strategy"])
    if "risk" in raw:
        cfg.risk = RiskConfig(**raw["risk"])
    if "email" in raw:
        cfg.email = EmailConfig(**raw["email"])
    if "broker" in raw:
        cfg.broker = BrokerConfig(**raw["broker"])

    return cfg
