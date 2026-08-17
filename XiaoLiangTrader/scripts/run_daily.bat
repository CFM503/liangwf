@echo off
chcp 65001 >nul
:: ============================================================================
:: XiaoLiangTrader 每日定时任务执行脚本 (Windows Batch)
:: 建议配合 Windows 任务计划程序 (Task Scheduler) 每天 15:10 自动触发
:: ============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%..\.."

:: 确保日志目录存在
if not exist "XiaoLiangTrader\logs" mkdir "XiaoLiangTrader\logs"

echo [%date% %time%] 🚀 开始执行 XiaoLiangTrader 每日买卖点扫描... >> "XiaoLiangTrader\logs\daily_task.log"

:: 执行 Python 扫描与通知 (同时支持邮件与钉钉)
python XiaoLiangTrader\scripts\daily_signal.py --notify --notify-dingtalk >> "XiaoLiangTrader\logs\daily_task.log" 2>&1

echo [%date% %time%] ✅ 扫描与通知执行完成 >> "XiaoLiangTrader\logs\daily_task.log"
