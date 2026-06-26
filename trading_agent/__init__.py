"""
TradingAgent — A股自动化交易系统
"""

from .agent import TradingAgent
from .config import TradingConfig, load_config
from .strategy import DualMAStrategy, Signal, Action
from .executor import SimulatorExecutor, create_executor
from .risk import RiskManager
from .notifier import Notifier
from .logger import log

__version__ = "1.0.0"
__all__ = [
    "TradingAgent",
    "TradingConfig",
    "load_config",
    "DualMAStrategy",
    "Signal",
    "Action",
    "SimulatorExecutor",
    "create_executor",
    "RiskManager",
    "Notifier",
    "log",
]
