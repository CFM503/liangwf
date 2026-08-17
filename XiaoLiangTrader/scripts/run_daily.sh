#!/bin/bash
# ============================================================================
# XiaoLiangTrader 每日定时任务执行脚本 (Linux / macOS)
# 建议配合 crontab 或 launchd 每天 15:10 自动触发
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$PROJECT_ROOT/XiaoLiangTrader/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 开始执行 XiaoLiangTrader 每日买卖点扫描..." >> "$PROJECT_ROOT/XiaoLiangTrader/logs/daily_task.log"

python3 "$PROJECT_ROOT/XiaoLiangTrader/scripts/daily_signal.py" --notify --notify-dingtalk >> "$PROJECT_ROOT/XiaoLiangTrader/logs/daily_task.log" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 扫描与通知执行完成" >> "$PROJECT_ROOT/XiaoLiangTrader/logs/daily_task.log"
