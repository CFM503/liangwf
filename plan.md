# 自动化交易 Agent 实现计划

## 架构

```
Liangwf/trading_agent/
├── __init__.py
├── config.py           # 配置管理（YAML + 加密）
├── crypto.py           # API Key AES加密/解密
├── data_feed.py        # 数据获取（复用 data_fetcher.py）
├── strategy.py         # 策略引擎（调用现有 DualMA）
├── executor.py         # 订单执行（模拟/实盘）
├── risk.py             # 风控模块 + 紧急停止
├── notifier.py         # Telegram + 邮件通知
├── logger.py           # 结构化日志
├── agent.py            # 主 Agent（调度编排）
├── main.py             # 入口 + schedule 定时
└── config.example.yaml # 配置模板（不含密钥）
```

## 模块职责

### config.py
- YAML 读取，支持环境变量覆盖
- 敏感字段（api_key, tg_token）读取时自动解密

### crypto.py
- Fernet（AES-128-CBC）对称加密
- `encrypt_config()` / `decrypt_value()` 接口
- 密钥从环境变量 `AGENT_SECRET` 或文件 `~/.trading_agent/secret.key` 读取

### strategy.py
- 复用 `strategy_dual_ma.py` 中的信号逻辑（不依赖 backtrader 运行时）
- 纯 pandas 计算：返回 `Signal(action, symbol, price, size)`

### executor.py
- `SimulatorExecutor`: 模拟成交，记录到 SQLite
- `LiveExecutor`: 预留接口（需券商API），当前 raise NotImplementedError
- 统一接口 `execute(signal) -> OrderResult`

### risk.py
- 紧急停止文件 `trading_agent/.kill_switch`，存在即停
- 也支持 Telegram `/stop` 命令远程停止
- 每日最大亏损、单笔限额检查

### notifier.py
- Telegram: `python-telegram-bot` 发消息
- 邮件: `smtplib` + SSL
- 统一 `notify(title, body, level)` 接口

### logger.py
- `logging` 模块，文件 + 控制台双输出
- 日志格式: `[2025-06-27 09:30:00] [INFO] [AGENT] ...`
- 按日期轮转，保留30天

### agent.py
- `TradingAgent.run_daily()` 主流程:
  1. 获取数据
  2. 计算信号
  3. 风控检查
  4. 执行订单
  5. 通知 + 日志

### main.py
- `schedule.every().day.at("15:10").do(agent.run_daily)` (A股收盘后)
- 支持 `--once` 单次运行、`--encrypt` 初始化加密
- Ctrl+C 优雅退出

## 关键设计

1. **不依赖 backtrader 运行时** — 策略信号用纯 pandas 计算，Agent 轻量化
2. **模拟器用 SQLite** — 订单、持仓、资金全部持久化，重启不丢状态
3. **Kill Switch** — 文件 + Telegram 双通道，0.5s 轮询
4. **密钥零明文** — config.yaml 中存密文，运行时解密
