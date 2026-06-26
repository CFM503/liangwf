# 🏫 XiaoLiangTrader — 校园股神量化系统

> "2008-2010，浙大宿舍，一台个人电脑，用编程自动化 + 传统机器学习在股市里找机会。"

一个面向个人学习的 A 股量化交易项目。
**双均线策略 + 成交量过滤 + LightGBM/XGBoost/sklearn 预测 + 全自动定时交易。**

不依赖大模型，不依赖 GPU，16GB 内存的笔记本就能跑。

**⚠️ 声明：仅供学习，不构成投资建议。股市有风险，入市需谨慎。**

---

## 📁 项目结构

```
XiaoLiangTrader/
├── main.py                  # 🚀 一键入口
├── config/
│   ├── config.yaml          # ⚙️ 配置文件
│   └── settings.py
├── data/
│   ├── fetcher.py           # 📊 akshare 数据获取 + CSV 缓存
│   └── cache/
├── strategy/
│   ├── signals.py           # 信号定义
│   ├── dual_ma.py           # 📈 双均线交叉策略
│   └── ml_strategy.py       # 🤖 ML 增强策略
├── ml_model/
│   ├── features.py          # 🧮 30 个技术指标特征
│   ├── predictor.py         # 🌲 统一 ML 预测器（LGB/XGB/sklearn）
│   └── saved_models/
├── backtest/
│   └── engine.py            # 📉 Backtrader 回测
├── bot/
│   ├── executor.py          # 💰 订单执行（模拟/实盘预留）
│   ├── risk.py              # 🛡️ 风控
│   ├── notifier.py          # 📧 邮件通知
│   └── scheduler.py         # 🤖 交易 Agent
├── utils/
│   ├── logger.py            # 📝 日志
│   └── crypto.py            # 🔐 加密
├── requirements.txt
└── README.md
```

## 🚀 快速开始

```bash
cd XiaoLiangTrader
pip install -r requirements.txt

# 回测
python main.py --backtest

# 训练 ML 模型
python main.py --train

# 手动跑一次
python main.py --once

# 启动每日自动交易（15:10 收盘后）
python main.py

# 紧急停止
python main.py --stop
```

## 🤖 ML 模型选择

在 `config/config.yaml` 中设置 `ml.model_type`：

| 模型 | 速度 | 精度 | 说明 |
|------|------|------|------|
| `lightgbm` | ⭐⭐⭐ | ⭐⭐⭐ | **推荐**，速度最快 |
| `xgboost` | ⭐⭐ | ⭐⭐⭐ | 精度略高，速度稍慢 |
| `random_forest` | ⭐⭐ | ⭐⭐ | 最稳定，不容易过拟合 |
| `gradient_boosting` | ⭐ | ⭐⭐ | sklearn 自带，兼容性好 |

全部 < 100MB 内存，7 年日线数据训练 < 5 秒。

## 🧮 特征工程（30 个技术指标）

| 类别 | 特征 |
|------|------|
| 均线 | MA5/10/20/60 偏离率，多空排列 |
| 动量 | 1/5/10/20 日收益率 |
| 波动 | 10/20 日波动率，ATR(14) |
| 成交量 | 5/20 日量比，量价相关性 |
| RSI | RSI(14) |
| MACD | DIF, DEA, MACD 柱 |
| KDJ | K, D, J 值 |
| 布林带 | 带宽，价格位置 |
| K线 | 上下影线，实体比，振幅 |
| 趋势 | 20 日线性回归斜率，收盘价位置 |

## ⚙️ 配置

编辑 `config/config.yaml`：

```yaml
stocks: ["600519", "300750", "601318"]

strategy:
  fast_period: 5
  slow_period: 20
  stop_loss: 0.08

ml:
  enabled: true              # 改为 true 启用
  model_type: "lightgbm"     # lightgbm / xgboost / random_forest / gradient_boosting
  confidence_threshold: 0.6
```

## 🛡️ 风控机制

| 机制 | 说明 |
|------|------|
| Kill Switch | 创建文件即停止，CLI 控制 |
| T+1 | 今日买入不能今日卖出 |
| 涨停不追 | ≥ 9.5% 不买入 |
| 跌停不卖 | ≤ -9.5% 不尝试卖出 |
| 止损 -8% | 强制卖出 |
| 跟踪止盈 | 盈利 15% 后回撤 5% 卖出 |
| 仓位限制 | 单股 ≤ 20%，总仓位 ≤ 60% |
| 日亏损限额 | 当日 ≥ 3% 暂停交易 |

## ⚠️ 风险警告

- **过拟合**：回测漂亮不代表实盘赚钱
- **滞后性**：MA 是滞后指标，金叉时行情可能已走一半
- **震荡市**：横盘期反复止损是最大风险
- **样本外验证**：2018-2022 训练，2023-2025 验证
- **纸上交易**：实盘前至少模拟 3 个月

---

**仅供学习，不构成投资建议。** 🎓
