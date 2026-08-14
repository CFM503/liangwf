"""
结构化日志 — 文件（按日轮转）+ 控制台双输出
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logger(
    name: str = "trading_agent",
    log_dir: str = "logs",
    level: int = logging.INFO,
    backup_count: int = 30,
) -> logging.Logger:
    """
    创建 logger，同时输出到文件和控制台

    Args:
        name: logger 名称
        log_dir: 日志目录
        level: 日志级别
        backup_count: 保留天数

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    # 防止重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # 日志格式
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 文件（按日轮转）
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        log_path / f"{name}.log",
        when="midnight",
        interval=1,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    file_handler.suffix = "%Y-%m-%d"
    logger.addHandler(file_handler)

    return logger


# 全局 logger 实例
log = setup_logger()
