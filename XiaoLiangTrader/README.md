# 🏫 XiaoLiangTrader — 校园股神量化系统

> "在宿舍里用一台笔记本，也能跑通量化交易的全流程。"

一个面向个人学习的 A 股量化交易项目。双均线策略 + 成交量过滤 + LightGBM 预测 + 本地大模型辅助 + 全自动定时交易。

**⚠️ 声明：本项目仅供学习，不构成任何投资建议。股市有风险，入市需谨慎。**

---

## 📁 项目结构

```
XiaoLiangTrader/
├── main.py                  # 🚀 一键入口（所有命令都在这）
├── config/
│   ├── config.yaml          # ⚙️ 配置文件（改参数不用改代码）
│   └── settings.py          # 配置加载
├── data/
│   ├── fetcher.py           # 📊 akshare 数据获取 + CSV 缓存
│   └── cache/               # 本地缓存
├── strategy/
│   ├── signals.py           # 信号定义（BUY/SELL/HOLD）
│   ├── dual_ma.py           # 📈 双均线交叉策略
│   └── ml_strategy.py       # 🤖 ML 增强策略（LightGBM + LLM）
├── ml_model/
│   ├── features.py          # 🧮 特征工程（20+ 技术指标）
│   ├── lgb_model.py         # 🌲 LightGBM 预测模型
│   ├── llm_advisor.py       # 🧠 本地大模型（Ollama/LM Studio）
│   └── saved_models/        # 训练好的模型
├── backtest/
│   └── engine.py            # 📉 Backtrader 回测引擎
├── bot/
│   ├── executor.py          # 💰 订单执行（模拟盘 SQLite / 实盘预留）
│   ├── risk.py              # 🛡️ 风控（Kill Switch / 涨跌停 / 仓位）
│   ├── notifier.py          # 📧 邮件通知
│   └── scheduler.py         # 🤖 交易 Agent（每日编排）
├── utils/
│   ├── logger.py            # 📝 日志（按日轮转）
│   └── crypto.py            # 🔐 API Key 加密
├── logs/                    # 日志文件
├── output/                  # 回测图表
└── requirements.txt
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd XiaoLiangTrader
pip install -r requirements.txt
```

### 2. 回测（推荐先跑这个）

```bash
python main.py --backtest
```

输出：年化收益率、胜率、最大回撤、夏普比率 + K线图（`output/backtest_result.png`）

### 3. 训练 ML 模型（可选）

```bash
# 需要 lightgbm
python main.py --train
```

### 4. 单次运行（测试）

```bash
python main.py --once
```

### 5. 启动每日自动交易

```bash
python main.py
# 每天 15:10（收盘后）自动运行
# Ctrl+C 退出
```

### 6. 紧急停止 / 恢复

```bash
python main.py --stop     # 立即停止所有交易
python main.py --resume   # 恢复
```

### 7. 查看账户状态

```bash
python main.py --status
```

## ⚙️ 配置说明

编辑 `config/config.yaml`：

```yaml
# 标的池
stocks: ["600519", "300750", "601318"]

# 策略参数
strategy:
  fast_period: 5       # 短期均线
  slow_period: 20      # 长期均线
  vol_mult: 1.5        # 成交量放大倍数
  stop_loss: 0.08      # 止损 8%
  take_profit: 0.15    # 止盈 15%

# ML 增强
ml:
  enabled: false       # 改为 true 启用

# 本地大模型
llm:
  enabled: false       # 改为 true 启用
  model_name: "qwen2.5:1.5b"
```

## 🤖 本地大模型接入

需要先安装 [Ollama](https://ollama.com)：

```bash
# 安装 Ollama 后拉取模型
ollama pull qwen2.5:1.5b

# 在 config.yaml 中启用
llm:
  enabled: true
```

LLM 的角色是**辅助决策**，不是独立交易。它读取技术指标后给出"买/不建议"的参考，最终由双均线策略和风控做决定。

## 🛡️ 风控机制

| 机制 | 说明 |
|------|------|
| Kill Switch | 文件 `.kill_switch` 存在即停止，CLI 命令控制 |
| T+1 | 今日买入不能今日卖出（A股规则） |
| 涨停不追 | 涨幅 ≥ 9.5% 不买入 |
| 跌停不卖 | 跌幅 ≤ -9.5% 不尝试卖出 |
| 止损 | 单笔亏损 -8% 强制卖出 |
| 跟踪止盈 | 盈利 15% 后从最高点回撤 5% 卖出 |
| 仓位限制 | 单股 ≤ 20%，总仓位 ≤ 60% |
| 日亏损限额 | 当日亏损 ≥ 3% 暂停交易 |
| 密钥加密 | `--encrypt` 加密 config.yaml 中的密码 |

## ⚠️ 风险警告

### 策略风险
- **滞后性**：MA 是滞后指标，金叉时行情可能已走一半
- **震荡市**：横盘期频繁假突破，反复止损（最大风险）
- **过拟合**：回测漂亮不代表实盘赚钱，参数可能只对历史有效
- **幸存者偏差**：选的都是好股票，不代表未来

### A 股特殊风险
- **涨跌停**：可能无法执行止损
- **T+1**：止损有 1 天延迟
- **印花税**：卖出千一，频繁交易成本高
- **滑点冲击**：小盘股尤其严重

### 改进建议
1. **趋势过滤**：加 MA60/MA120，只在上升趋势做多
2. **ATR 动态止损**：用 ATR×2 代替固定止损
3. **样本外验证**：2018-2022 训练，2023-2025 验证
4. **Walk-forward**：滚动窗口优化，避免过拟合
5. **行业轮动**：按 ETF 轮动而非个股
6. **纸上交易**：实盘前至少模拟 3 个月

---

**仅供学习，不构成投资建议。** 祝你在量化之路上越走越远 🎓
