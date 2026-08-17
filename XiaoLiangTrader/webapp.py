"""
XiaoLiangTrader — 本地量化交易与信号看板 (Streamlit Web UI)
============================================================
启动命令:
    streamlit run XiaoLiangTrader/webapp.py

浏览器自动访问:
    http://localhost:8501

功能说明:
1. 🎯 今日买卖点预测看板：直观展示每日买卖信号、止损止盈位与 ML 置信度
2. 📋 实盘信号历史留痕：查看已持久化记录的历史信号日志
3. 📈 实盘事后复盘检验：面向未来的真实跟踪复盘（胜率、平均收益、盈亏比）
4. 💰 2万小资金可买性清单：60只股票池在 2 万元本金（单股上限4000元）下的可买性分布
5. 📊 2万本金可行性压力测试：横向对比 100万 vs 2万全池 vs 2万低价池的 3 折时序验证表现
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
import pandas as pd
import numpy as np

from scripts.review_live_signals import LiveSignalReviewer
from data.fetcher import STOCK_NAMES

RESULTS_DIR = _root / "ml_model" / "eval_results"


# ══════════════════════════════════════════════════════════════
# Streamlit 页面配置与主题
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="XiaoLiangTrader 量化看板",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 20px;
    }
    .stat-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .stat-val {
        font-size: 22px;
        font-weight: bold;
        color: #0F172A;
    }
    .stat-lbl {
        font-size: 12px;
        color: #64748B;
    }
    .badge-buy {
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-sell {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 数据加载辅助函数
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_latest_signals() -> dict:
    file_path = RESULTS_DIR / "daily_signals_latest.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"读取 daily_signals_latest.json 失败: {e}")
    return {}


@st.cache_data(ttl=60)
def load_live_log() -> pd.DataFrame:
    file_path = RESULTS_DIR / "live_signals_log.csv"
    if file_path.exists() and os.path.getsize(file_path) > 0:
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype={"symbol": str})
            return df
        except Exception:
            try:
                df = pd.read_csv(file_path, encoding="utf-8", dtype={"symbol": str})
                return df
            except Exception:
                pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_affordability_data() -> dict:
    file_path = RESULTS_DIR / "affordability_20k.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@st.cache_data(ttl=60)
def load_wf_results() -> dict:
    results = {}
    for key, filename in [
        ("1m_60stocks", "walk_forward_60stocks.json"),
        ("20k_capital", "walk_forward_20k_capital.json"),
        ("20k_affordable", "walk_forward_20k_affordable_only.json"),
    ]:
        p = RESULTS_DIR / filename
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    results[key] = json.load(f)
            except Exception:
                results[key] = {}
        else:
            results[key] = {}
    return results


# ══════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏫 XiaoLiangTrader")
    st.caption("A股短线量化交易系统 (只读仪表盘)")
    st.divider()

    st.subheader("⚙️ 系统运行状态")
    st.markdown("**核心模型**: `LightGBM (ML过滤)`")
    st.markdown("**规则引擎**: `Dual MA (5/20) + 放量1.5x`")
    st.markdown("**初始本金**: `20,000 元` (小资金模式)")
    st.markdown("**单股风控**: `20% (上限 4,000 元/股)`")
    st.markdown("**卖出印花税**: `0.05% (千0.5)`")
    st.markdown("**通知渠道**: `邮件 (SMTP) / 钉钉多维表格`")

    st.divider()
    st.caption("💡 提示：本系统所有页面为纯只读展示，不涉及实盘自动下单或敏感配置修改。")


# ══════════════════════════════════════════════════════════════
# 主看板区域
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🎯 XiaoLiangTrader 量化策略与实盘跟踪看板</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">实时汇聚每日买卖点预测、实盘留痕日志、事后复盘检验、2万元小资金可买性与时序验证表现</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🎯 今日买卖点预测",
    "📋 实盘信号历史留痕",
    "📈 实盘事后复盘检验",
    "💰 2万小资金可买性清单",
    "📊 2万本金可行性压力测试",
])


# ──────────────────────────────────────────────────────────────
# Tab 1: 🎯 今日买卖点预测看板
# ──────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🎯 最新每日买卖点预测看板")
    latest_data = load_latest_signals()

    if not latest_data:
        st.info("💡 尚未生成每日买卖点信号，请在终端运行: `python XiaoLiangTrader/scripts/daily_signal.py`")
    else:
        # 指标卡片
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("扫描日期", latest_data.get("date", "-"))
        with col2:
            st.metric("已扫描标的", f"{latest_data.get('scanned_count', 0)} 只")
        with col3:
            st.metric("🟢 建议买入", f"{latest_data.get('buy_count', 0)} 只")
        with col4:
            st.metric("🔴 建议卖出", f"{latest_data.get('sell_count', 0)} 只")
        with col5:
            st.metric("⚪ 观望标的", f"{latest_data.get('no_signal_count', 0)} 只")

        signals = latest_data.get("signals", [])
        if not signals:
            st.success("✅ 今日扫描完成，无明确触发买卖点的标的（全市场处于震荡或持仓观望状态）。")
        else:
            table_rows = []
            for s in signals:
                act = s.get("action", "")
                act_tag = "🟢 建议买入" if act == "BUY" else "🔴 建议卖出"
                sl = f"{s['stop_loss_price']:.2f}" if s.get("stop_loss_price", 0) > 0 else "-"
                tp = f"{s['take_profit_price']:.2f}" if s.get("take_profit_price", 0) > 0 else "-"
                table_rows.append({
                    "股票代码": str(s.get("symbol", "")).zfill(6),
                    "股票名称": s.get("name", ""),
                    "建议动作": act_tag,
                    "最新价 (元)": f"{s.get('price', 0.0):.2f}",
                    "建议止损 (元)": sl,
                    "建议止盈 (元)": tp,
                    "ML置信度": f"{s.get('ml_confidence', 0.0):.3f}",
                    "触发原因": s.get("reason", ""),
                })
            df_display = pd.DataFrame(table_rows)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.caption(f"⚠️ {latest_data.get('disclaimer', '免责声明：本信号仅供量化模型学习与学术研究参考，不构成任何实际投资建议。')}")


# ──────────────────────────────────────────────────────────────
# Tab 2: 📋 实盘信号历史留痕
# ──────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("📋 实盘信号历史持久化留痕日志")
    df_live = load_live_log()

    if df_live.empty:
        st.info("💡 当前暂无历史留痕信号记录。每次运行 `daily_signal.py` 时会自动将信号追加写入 `live_signals_log.csv`。")
    else:
        st.markdown(f"**累计记录信号**: `{len(df_live)}` 条（数据源: `XiaoLiangTrader/ml_model/eval_results/live_signals_log.csv`）")
        
        # 过滤器
        c1, c2 = st.columns([2, 2])
        with c1:
            action_filter = st.selectbox("筛选动作", ["全部动作", "BUY", "SELL"])
        with c2:
            search_query = st.text_input("按代码或名称搜索", "")

        df_filtered = df_live.copy()
        if action_filter != "全部动作":
            df_filtered = df_filtered[df_filtered["action"] == action_filter]
        if search_query:
            df_filtered = df_filtered[
                df_filtered["symbol"].str.contains(search_query, na=False) |
                df_filtered["name"].str.contains(search_query, na=False)
            ]

        df_filtered = df_filtered.sort_values(by="date", ascending=False)
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Tab 3: 📈 实盘事后复盘检验 (面向未来真实跟踪)
# ──────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 实盘信号事后复盘检验 (面向未来真实跟踪)")
    st.markdown("自动读取累积记录的实盘信号，回填后续真实 K 线（1~5日），统计真实胜率、平均收益与止损止盈触发情况，**与历史回测数据严格物理隔离**。")

    reviewer = LiveSignalReviewer()
    df_for_review = reviewer.load_signals(use_demo=False)
    
    if df_for_review.empty or len(df_for_review) == 0:
        st.info("💡 当前暂无累积实盘信号，为您展示历史内置示例看板：")
        df_for_review = reviewer.load_signals(use_demo=True)

    summary = reviewer.evaluate_signals(df_for_review)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("累计记录信号", f"{summary.get('total_signals', 0)} 条")
    with m2:
        st.metric("已结项买入", f"{summary.get('completed_buy_signals', 0)} 笔")
    with m3:
        st.metric("实盘胜率", f"{summary.get('win_rate', 0.0):.1f}%")
    with m4:
        st.metric("平均单笔收益", f"{summary.get('avg_return_pct', 0.0):+.2f}%")
    with m5:
        st.metric("盈亏比", f"{summary.get('profit_loss_ratio', 0.0):.2f}")

    reviews_list = summary.get("reviews", [])
    if reviews_list:
        rev_rows = []
        for r in reviews_list:
            act_label = "🟢 买入" if r["action"] == "BUY" else "🔴 卖出"
            sl = f"{r['stop_loss']:.2f}" if r['stop_loss'] > 0 else "-"
            tp = f"{r['take_profit']:.2f}" if r['take_profit'] > 0 else "-"
            mg = f"{r['max_gain_pct']:+.2f}%" if r["action"] == "BUY" else "-"
            ret = f"{r['exit_return_pct']:+.2f}%"
            rev_rows.append({
                "信号日期": r["date"],
                "代码": r["symbol"],
                "名称": r["name"],
                "动作": act_label,
                "触发价 (元)": f"{r['entry_price']:.2f}",
                "建议止损": sl,
                "建议止盈": tp,
                "持股天数": f"{r['holding_days']} 天",
                "最高涨幅": mg,
                "实际收益": ret,
                "复盘状态": f"{r['status']} ({r['exit_reason']})",
            })
        st.dataframe(pd.DataFrame(rev_rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Tab 4: 💰 2万小资金可买性清单
# ──────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("💰 2 万元小资金标的可买性检验清单")
    aff_data = load_affordability_data()

    if not aff_data:
        st.info("💡 尚未生成可买性数据，请在终端运行: `python XiaoLiangTrader/scripts/check_affordability.py`")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("初始本金", f"{aff_data.get('initial_cash', 20000):,.0f} 元")
        with k2:
            st.metric("单股仓位上限", f"{aff_data.get('max_single_pct', 0.2)*100:.0f}% (最多 {aff_data.get('max_single_cash_limit', 4000):,.0f} 元/股)")
        with k3:
            st.metric("🟢 可买股票 (≤40元)", f"{aff_data.get('affordable_count', 0)} 只 ({aff_data.get('affordable_ratio', '-')})")
        with k4:
            st.metric("🔴 买不起股票 (>40元)", f"{aff_data.get('unaffordable_count', 0)} 只 ({aff_data.get('unaffordable_ratio', '-')})")

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown(f"#### 🟢 可买股票清单 (共 {aff_data.get('affordable_count', 0)} 只)")
            st.caption("一手（100股）成本 ≤ 4,000 元，小资金可正常建仓")
            aff_list = aff_data.get("affordable_stocks", [])
            if aff_list:
                df_aff = pd.DataFrame(aff_list)[["symbol", "name", "latest_price", "one_lot_cost"]]
                df_aff.columns = ["代码", "名称", "最新股价 (元)", "1手建仓成本 (元)"]
                st.dataframe(df_aff, use_container_width=True, hide_index=True)

        with sub_col2:
            st.markdown(f"#### 🔴 买不起高价股票清单 (共 {aff_data.get('unaffordable_count', 0)} 只)")
            st.caption("一手（100股）成本 > 4,000 元，20% 单股风控下无法买入 1 手")
            unaff_list = aff_data.get("unaffordable_stocks", [])
            if unaff_list:
                df_unaff = pd.DataFrame(unaff_list)[["symbol", "name", "latest_price", "one_lot_cost", "shortfall_per_lot"]]
                df_unaff.columns = ["代码", "名称", "最新股价 (元)", "1手建仓成本 (元)", "超额差额 (元)"]
                st.dataframe(df_unaff, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Tab 5: 📊 2万本金可行性压力测试
# ──────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("📊 2万元本金可行性压力测试与 3 折 Walk-Forward 对比")
    st.markdown("基于 Expanding Window 时序递增滚动验证，评估在真实 A 股 T+1、涨跌停、滑点与 **卖出 0.05% 印花税** 下，不同资金规模与股票池的真实表现：")

    wf_data = load_wf_results()

    # 对比汇总表格
    comp_rows = [
        {
            "实验场景": "1. 100万本金 (60只全池, 含印花税)",
            "2022 熊市": "+6.35% (29笔)",
            "2023 震荡市": "-0.74% (35笔)",
            "2024 反弹市": "+8.10% (28笔)",
            "3折总交易": "92 笔",
            "平均胜率": "39.9%",
            "3折平均年化": "+4.57%",
            "策略状态": "🟢 稳健正收益",
        },
        {
            "实验场景": "2. 2万本金 (60只全池, 单股上限4000元)",
            "2022 熊市": "-1.14% (21笔)",
            "2023 震荡市": "-12.84% (23笔)",
            "2024 反弹市": "+14.98% (15笔)",
            "3折总交易": "59 笔",
            "平均胜率": "33.9%",
            "3折平均年化": "+0.34%",
            "策略状态": "🟡 勉强打平 (样本缩水35.9%)",
        },
        {
            "实验场景": "3. 2万本金 (仅26只<=40元低价池)",
            "2022 熊市": "-4.76% (15笔)",
            "2023 震荡市": "-15.76% (20笔)",
            "2024 反弹市": "+4.20% (8笔)",
            "3折总交易": "43 笔",
            "平均胜率": "31.1%",
            "3折平均年化": "-5.44%",
            "策略状态": "🔴 转负亏损 (低价股动量缺失)",
        },
    ]

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💡 核心客观结论与方案权衡")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 方案 A：提高单股上限 `max_single_pct`（集中持仓）")
        st.markdown("""
        - **参数调整**：调至 `33%` (单股 6,600 元，可买 40 只) 或 `50%` (单股 10,000 元，可买 49 只)。
        - **🟢 优势**：重新覆盖宁德时代、比亚迪、美的集团、立讯精密等高成长景气龙头，保留策略 Alpha 与交易机会。
        - **🔴 风险**：**个股集中度风险剧烈放大**。持仓仅 2~3 只股票，单股若触及 -8% 止损对总资产造成 **-2.6% ~ -4.0%** 穿透损失，最大回撤显著加剧。
        """)

    with col_b:
        st.markdown("### 方案 B：专门筛选中低价股票池（现价 ≤ 40 元）")
        st.markdown("""
        - **参数调整**：保持 20% 分散度（配置 5 只股票，每只 4,000 元），但限定低价池。
        - **🟢 优势**：维持 5 只股票的分散度，单笔 -8% 止损仅损失总资产 1.6%，控制单股黑天鹅。
        - **🔴 风险**：**股票池品质严重退化（低价股负向选择偏误）**。A 股低价股多为重资产传统周期、金融地产与滞涨股，实测 3 折年化由 **+0.34% 恶化为 -5.44%**，反弹市缺乏动量。
        """)
