"""
配置管理 — 读取 YAML，自动解密敏感字段
=======================================
所有参数集中管理，改参数不用改代码。
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field

from utils.crypto import maybe_decrypt

# 默认配置文件路径
_CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"


@dataclass
class StrategyParams:
    """策略参数"""
    fast_period: int = 5           # 短期均线周期
    slow_period: int = 20          # 长期均线周期
    vol_period: int = 20           # 均量计算周期
    vol_mult: float = 1.5          # 成交量放大倍数阈值
    stop_loss: float = 0.08        # 止损比例（8%）
    take_profit: float = 0.15      # 止盈比例（15%）
    trailing_pct: float = 0.05     # 跟踪止盈回撤比例（5%）


@dataclass
class RiskParams:
    """风控参数"""
    max_single_pct: float = 0.20   # 单股最大仓位占比
    max_total_pct: float = 0.60    # 总仓位上限
    max_daily_loss_pct: float = 0.03  # 每日最大亏损
    limit_up_threshold: float = 0.095  # 涨停阈值（9.5%，接近10%不追）
    limit_down_threshold: float = -0.095  # 跌停阈值
    kill_switch_file: str = ".kill_switch"


@dataclass
class MLParams:
    """机器学习参数"""
    enabled: bool = False          # 是否启用 ML 增强
    lookback_days: int = 60        # 特征回看天数
    retrain_days: int = 30         # 每 N 天重新训练
    confidence_threshold: float = 0.6  # ML 信号置信度阈值


@dataclass
class LLMParams:
    """本地大模型参数"""
    enabled: bool = False          # 是否启用 LLM 辅助
    api_url: str = "http://localhost:11434/api/generate"  # Ollama 默认地址
    model_name: str = "qwen2.5:1.5b"  # 模型名
    timeout: int = 30              # 请求超时（秒）


@dataclass
class EmailParams:
    """邮件通知参数"""
    enabled: bool = False
    smtp_server: str = "smtp.qq.com"
    smtp_port: int = 465
    sender: str = ""
    password: str = ""  # 加密存储
    receiver: str = ""


@dataclass
class BrokerParams:
    """券商参数"""
    mode: str = "simulator"  # simulator / live
    api_key: str = ""
    api_secret: str = ""
    account_id: str = ""


@dataclass
class Config:
    """总配置"""
    # 标的池
    stocks: list[str] = field(default_factory=lambda: ["600519", "300750", "601318"])
    stock_names: dict = field(default_factory=lambda: {
        "600519": "贵州茅台", "300750": "宁德时代", "601318": "中国平安",
        "000858": "五粮液", "600036": "招商银行",
    })
    # 资金
    initial_cash: float = 1_000_000
    # 定时运行时间（A股收盘后）
    run_time: str = "15:10"
    # 数据回溯起始日
    data_start: str = "20180101"
    # 子模块参数
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    ml: MLParams = field(default_factory=MLParams)
    llm: LLMParams = field(default_factory=LLMParams)
    email: EmailParams = field(default_factory=EmailParams)
    broker: BrokerParams = field(default_factory=BrokerParams)
    # 路径
    log_dir: str = "logs"
    db_path: str = "xlt.db"


def load_config(path: str | Path | None = None) -> Config:
    """
    加载 YAML 配置文件。
    文件不存在时返回默认配置。
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not path.exists():
        print(f"[config] 未找到 {path}，使用默认参数")
        return Config()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    # 顶层字段
    cfg.stocks = raw.get("stocks", cfg.stocks)
    cfg.stock_names = raw.get("stock_names", cfg.stock_names)
    cfg.initial_cash = raw.get("initial_cash", cfg.initial_cash)
    cfg.run_time = raw.get("run_time", cfg.run_time)
    cfg.data_start = raw.get("data_start", cfg.data_start)
    cfg.log_dir = raw.get("log_dir", cfg.log_dir)
    cfg.db_path = raw.get("db_path", cfg.db_path)

    # 子模块配置
    for section, cls in [
        ("strategy", StrategyParams),
        ("risk", RiskParams),
        ("ml", MLParams),
        ("llm", LLMParams),
    ]:
        if section in raw:
            setattr(cfg, section, cls(**{k: v for k, v in raw[section].items()
                                         if k in cls.__dataclass_fields__}))

    # 需要解密的模块
    for section_name, cls in [("email", EmailParams), ("broker", BrokerParams)]:
        if section_name in raw:
            decrypted = {}
            for k, v in raw[section_name].items():
                decrypted[k] = maybe_decrypt(v) if isinstance(v, str) else v
            setattr(cfg, section_name, cls(**{k: v for k, v in decrypted.items()
                                               if k in cls.__dataclass_fields__}))

    return cfg
