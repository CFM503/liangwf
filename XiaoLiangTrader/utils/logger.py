"""
日志工具 — 文件（按日轮转）+ 控制台双输出
==========================================
宿舍里调试策略，日志是最靠谱的伙伴。
"""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def get_logger(
    name: str = "xlt",
    log_dir: str = "logs",
    level: int = logging.INFO,
    backup_days: int = 30,
) -> logging.Logger:
    """
    获取一个 logger 实例，同时输出到文件和控制台。

    Args:
        name: logger 名称（建议用模块名，如 'xlt.strategy'）
        log_dir: 日志文件存放目录
        level: 日志级别
        backup_days: 日志保留天数

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    # 防止重复添加 handler（很重要！）
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # 日志格式：[时间] [级别] [模块] 消息
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 控制台输出 ──
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── 文件输出（按日轮转）──
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = TimedRotatingFileHandler(
        log_path / f"{name}.log",
        when="midnight",       # 每天午夜轮转
        interval=1,
        backupCount=backup_days,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    file_handler.suffix = "%Y-%m-%d"
    logger.addHandler(file_handler)

    return logger


# 全局默认 logger
log = get_logger()
