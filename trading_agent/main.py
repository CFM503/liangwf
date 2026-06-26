#!/usr/bin/env python3
"""
交易 Agent 入口 — schedule 定时调度 + CLI 命令

用法:
    python -m trading_agent.main                  # 启动定时调度（每日运行）
    python -m trading_agent.main --once           # 单次运行
    python -m trading_agent.main --status         # 查看状态
    python -m trading_agent.main --encrypt        # 加密配置文件中的敏感字段
    python -m trading_agent.main --stop           # 激活 Kill Switch
    python -m trading_agent.main --resume         # 解除 Kill Switch
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_agent.config import load_config
from trading_agent.agent import TradingAgent
from trading_agent.risk import RiskManager
from trading_agent.crypto import encrypt_config_file
from trading_agent.logger import log


# 敏感字段列表
ENCRYPT_FIELDS = ["password", "api_key", "api_secret", "token"]


def cmd_run_scheduler(config_path: str | None = None):
    """启动定时调度"""
    try:
        import schedule
    except ImportError:
        print("请安装 schedule: pip install schedule")
        sys.exit(1)

    config = load_config(config_path)
    agent = TradingAgent(config=config)
    run_time = config.run_time

    log.info("=" * 60)
    log.info("[main] TradingAgent 定时调度启动")
    log.info(f"[main] 每日运行时间: {run_time}")
    log.info(f"[main] 标的: {config.stocks}")
    log.info(f"[main] 模式: {config.broker.mode}")
    log.info("[main] Ctrl+C 优雅退出")
    log.info("=" * 60)

    schedule.every().day.at(run_time).do(agent.run_daily)

    # 优雅退出
    def graceful_exit(signum, frame):
        log.info("[main] 收到退出信号，正在关闭...")
        agent.risk.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    while True:
        schedule.run_pending()
        time.sleep(30)  # 每 30 秒检查一次


def cmd_run_once(config_path: str | None = None):
    """单次运行"""
    config = load_config(config_path)
    agent = TradingAgent(config=config)
    agent.run_daily()


def cmd_status(config_path: str | None = None):
    """查看状态"""
    import json

    config = load_config(config_path)
    agent = TradingAgent(config=config)
    status = agent.get_status()
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))


def cmd_encrypt(config_path: str):
    """加密配置文件"""
    print(f"[main] 正在加密配置文件: {config_path}")
    encrypt_config_file(config_path, ENCRYPT_FIELDS)


def cmd_stop():
    """激活 Kill Switch"""
    risk = RiskManager()
    risk.activate_kill_switch()
    print("🚨 Kill Switch 已激活！所有交易已停止。")
    print(f"   删除 {risk.kill_switch_file} 或运行 --resume 解除")


def cmd_resume():
    """解除 Kill Switch"""
    risk = RiskManager()
    risk.deactivate_kill_switch()
    print("✅ Kill Switch 已解除，交易恢复正常。")


def main():
    parser = argparse.ArgumentParser(
        description="TradingAgent — A股自动化交易Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m trading_agent.main                  # 启动定时调度
  python -m trading_agent.main --once           # 立即执行一次
  python -m trading_agent.main --status         # 查看账户状态
  python -m trading_agent.main --encrypt        # 加密配置
  python -m trading_agent.main --stop           # 紧急停止
  python -m trading_agent.main --resume         # 恢复交易
        """,
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--once", action="store_true", help="单次运行")
    parser.add_argument("--status", action="store_true", help="查看状态")
    parser.add_argument("--encrypt", action="store_true", help="加密配置文件")
    parser.add_argument("--stop", action="store_true", help="激活 Kill Switch（紧急停止）")
    parser.add_argument("--resume", action="store_true", help="解除 Kill Switch")

    args = parser.parse_args()

    if args.encrypt:
        config_path = args.config or str(Path(__file__).parent / "config.yaml")
        cmd_encrypt(config_path)
    elif args.stop:
        cmd_stop()
    elif args.resume:
        cmd_resume()
    elif args.status:
        cmd_status(args.config)
    elif args.once:
        cmd_run_once(args.config)
    else:
        cmd_run_scheduler(args.config)


if __name__ == "__main__":
    main()
