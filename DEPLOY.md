# 🚀 XiaoLiangTrader 操作系统级后台定时服务部署指南

> 本指南用于将每日买卖点扫描与通知任务（`daily_signal.py`）注册为**操作系统原生后台服务**。  
> 注册后，**无需保持命令行窗口常驻**，电脑开机后会在每个交易日下午 **15:10** 自动静默运行，并自动更新数据文件与推送通知。

---

## 💻 您的当前系统识别

根据当前环境路径（`D:\SOFT\AI\github\liangwf`），您当前运行的是 **Windows 操作系统**。  
推荐优先查阅 **第一部分：Windows 部署教程**。

---

## 1. 🪟 Windows 系统部署（两种方式，任选其一）

### 方式 A：PowerShell 一键自动注册（推荐，耗时 5 秒）

在 PowerShell（建议以管理员身份打开）中直接运行项目内置的注册脚本：

```powershell
powershell -ExecutionPolicy Bypass -File XiaoLiangTrader\scripts\register_windows_task.ps1
```

> **执行效果**：
> - 自动在 Windows 任务计划程序中创建名为 `XiaoLiangTrader_DailySignal` 的定时任务。
> - 设置为**每周一至周五（交易日）下午 15:10** 自动触发。
> - 任务日志将自动写入 `XiaoLiangTrader\logs\daily_task.log`。

---

### 方式 B：图形界面手动配置（手把手图文步骤）

如果你更习惯使用 Windows 可视化界面，请按照以下步骤操作：

1. **打开任务计划程序**：
   - 按下键盘快捷键 `Win + R`，输入 `taskschd.msc` 并回车。
2. **创建基本任务**：
   - 在右侧操作面板中，点击 **“创建基本任务...”**。
   - **名称**：填写 `XiaoLiangTrader_DailySignal`，点击“下一步”。
3. **设置触发器**：
   - 选择 **“每周”**，点击“下一步”。
   - **开始时间**：设置为今天或任意日期，时间设为 `15:10:00`。
   - **勾选天数**：勾选 `周一`、`周二`、`周三`、`周四`、`周五`（A股交易日），点击“下一步”。
4. **设置操作**：
   - 选择 **“启动程序”**，点击“下一步”。
   - **程序或脚本**：点击浏览，选择项目根目录下的执行脚本：
     `D:\SOFT\AI\github\liangwf\XiaoLiangTrader\scripts\run_daily.bat`
   - **起始于（可选）**（⚠️非常重要）：填写项目根目录绝对路径：
     `D:\SOFT\AI\github\liangwf`
   - 点击“下一步”，然后点击“完成”。
5. **测试运行**：
   - 在中间的任务列表中找到 `XiaoLiangTrader_DailySignal`，右键点击 **“运行”**。
   - 打开 `XiaoLiangTrader\logs\daily_task.log` 查看日志，确认有正常扫描输出即表示配置成功！

---

### 💡 Windows 进阶：静默无黑框弹窗运行

如果希望在 15:10 运行时**完全不弹出任何命令行黑色窗口**，项目内置了静默包装器 `run_silent.vbs`：

1. 在 `XiaoLiangTrader\scripts\` 目录下确认存在 `run_silent.vbs`：
   ```vbs
   Set WshShell = CreateObject("WScript.Shell")
   WshShell.Run "cmd /c XiaoLiangTrader\scripts\run_daily.bat", 0, False
   ```
2. 将任务计划程序里的“程序或脚本”改成 `wscript.exe`，“添加参数”填入 `XiaoLiangTrader\scripts\run_silent.vbs`。

---

## 2. 🍎 macOS 系统部署（launchd）

在 macOS 上，推荐使用系统级守护进程管理器 `launchd`：

1. **复制配置文件**：
   将 `XiaoLiangTrader/scripts/com.xiaoliangtrader.daily_signal.plist` 复制到用户的 LaunchAgents 目录：
   ```bash
   cp XiaoLiangTrader/scripts/com.xiaoliangtrader.daily_signal.plist ~/Library/LaunchAgents/
   ```

2. **修改实际路径**：
   打开 `~/Library/LaunchAgents/com.xiaoliangtrader.daily_signal.plist`，将其中的 `/path/to/liangwf` 替换为你的实际项目路径（如 `/Users/yourname/liangwf`）。

3. **加载并启动任务**：
   ```bash
   launchctl load ~/Library/LaunchAgents/com.xiaoliangtrader.daily_signal.plist
   ```

4. **常用管理命令**：
   - 卸载/停用任务：`launchctl unload ~/Library/LaunchAgents/com.xiaoliangtrader.daily_signal.plist`
   - 查看运行日志：`cat XiaoLiangTrader/logs/daily_task.log`

---

## 3. 🐧 Linux 系统部署（crontab）

在 Linux 服务器（如 Ubuntu / CentOS / Debian）上，使用原生的 `crontab`：

1. **打开定时任务编辑器**：
   ```bash
   crontab -e
   ```

2. **添加定时条目（每个交易日 15:10 执行）**：
   ```cron
   # XiaoLiangTrader 每日收盘后扫描 (周一到周五 15:10)
   10 15 * * 1-5 /usr/bin/bash /path/to/liangwf/XiaoLiangTrader/scripts/run_daily.sh >> /path/to/liangwf/XiaoLiangTrader/logs/daily_task.log 2>&1
   ```
   > ⚠️ 注意：请将 `/path/to/liangwf` 替换为你服务器上的真实绝对路径，并确保使用 `which python3` 确认 python 解释器路径。

3. **检查时区**：
   确保服务器时区为北京时间（`Asia/Shanghai`）：
   ```bash
   timedatectl set-timezone Asia/Shanghai
   ```

---

## 4. 🌐 与 Web UI 看板的联动效果

完成上述系统级任务注册后：
1. **无需人工干预**：每个交易日 15:10，后台服务会自动执行 `daily_signal.py`，完成全市场 60 只标的扫描、留痕追加写入 `live_signals_log.csv`，并通过邮件/钉钉推送。
2. **随时随地打开 Web 看板**：
   随时在终端执行：
   ```bash
   python -m streamlit run XiaoLiangTrader/webapp.py
   ```
   打开网页 `http://localhost:8501` 即可直接看到最新盘后自动计算好的买卖点预测与实盘跟踪，真正做到无人值守自动化！
