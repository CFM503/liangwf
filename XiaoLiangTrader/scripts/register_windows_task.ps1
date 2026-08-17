# ============================================================================
# XiaoLiangTrader 自动注册 Windows 计划任务脚本 (PowerShell)
# 作用: 在 Windows 任务计划程序中创建每天 15:10 自动运行信号扫描的任务
# 运行方式: 以管理员身份或普通用户在 PowerShell 中运行本脚本
# ============================================================================

$TaskName = "XiaoLiangTrader_DailySignal"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."
$BatPath = "$ScriptDir\run_daily.bat"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "🚀 正在注册 Windows 任务计划程序: $TaskName" -ForegroundColor Cyan
Write-Host "   项目路径: $ProjectRoot" -ForegroundColor Gray
Write-Host "   执行脚本: $BatPath" -ForegroundColor Gray
Write-Host "   触发时间: 交易日 (周一至周五) 15:10" -ForegroundColor Gray
Write-Host "================================================================" -ForegroundColor Cyan

# 检查任务是否已存在，若存在先删除
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "⚠️ 检测到已存在同名任务，正在更新..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# 创建 Action (执行 bat)
$Action = New-ScheduledTaskAction -Execute "$BatPath" -WorkingDirectory "$ProjectRoot"

# 创建 Trigger (每周一至周五 15:10 触发)
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "15:10"

# 创建 Settings (允许按需运行，错过后尽快补跑，电池供电也允许运行)
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

# 注册任务
try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "XiaoLiangTrader A股短线量化盘后自动扫描与通知任务"
    Write-Host "`n✅ 任务注册成功！" -ForegroundColor Green
    Write-Host "👉 您可以在 Windows '任务计划程序' 中搜索 '$TaskName' 查看或测试运行。" -ForegroundColor Green
} catch {
    Write-Host "`n❌ 任务注册失败: $_" -ForegroundColor Red
    Write-Host "💡 提示: 您可以使用管理员权限重新打开 PowerShell 再次运行，或者参考 DEPLOY.md 使用图形界面手动创建。" -ForegroundColor Yellow
}
