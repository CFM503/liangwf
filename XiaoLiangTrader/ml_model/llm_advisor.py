"""
本地大模型辅助决策 — 调用 Ollama / LM Studio
==============================================
不是用 AI 炒股！是让本地小模型读取技术指标，
给出一个"买/不建议"的参考意见。

需要本地运行 Ollama 或 LM Studio。
安装 Ollama: https://ollama.com
拉取模型: ollama pull qwen2.5:1.5b
"""

import json
import requests
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from utils.logger import get_logger

log = get_logger("xlt.llm")


@dataclass
class Opinion:
    """LLM 的建议"""
    should_buy: bool
    confidence: str   # "high", "medium", "low"
    text: str         # 原始回复文本


class LLMAdvisor:
    """
    本地大模型辅助顾问

    通过 HTTP API 调用本地运行的 Ollama 或 LM Studio，
    让小模型读取技术指标数据，给出交易建议。
    """

    def __init__(
        self,
        api_url: str = "http://localhost:11434/api/generate",
        model_name: str = "qwen2.5:1.5b",
        timeout: int = 30,
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout

    def _call_llm(self, prompt: str) -> str:
        """调用本地 LLM API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # 低温度，减少随机性
                "num_predict": 200,  # 限制输出长度
            },
        }
        try:
            resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"无法连接到 {self.api_url}\n"
                "请确保 Ollama 正在运行: ollama serve\n"
                "或 LM Studio 已启动本地服务器"
            )
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {e}")

    def ask_opinion(
        self,
        df: pd.DataFrame,
        symbol: str,
        signal=None,
    ) -> Opinion:
        """
        请 LLM 分析当前行情，给出买/不建议。

        Args:
            df: 含 OHLCV 的 DataFrame（取最近 10 天数据发给 LLM）
            symbol: 股票代码
            signal: 基础策略的 Signal 对象（可选）

        Returns:
            Opinion 对象
        """
        # 准备最近 10 天的数据摘要
        recent = df.tail(10)[["open", "high", "low", "close", "volume"]].copy()
        recent = recent.round(2)
        data_str = recent.to_string()

        # 计算简单指标
        last_close = df["close"].iloc[-1]
        ma5 = df["close"].tail(5).mean()
        ma20 = df["close"].tail(20).mean()
        vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(20).mean()

        prompt = f"""你是一个A股技术分析助手。请分析以下股票的近期走势，判断现在是否适合买入。

股票代码: {symbol}
当前价格: {last_close:.2f}
5日均线: {ma5:.2f}
20日均线: {ma20:.2f}
成交量比: {vol_ratio:.2f}倍

最近10天数据:
{data_str}

请用以下格式回答（不要解释太多）:
建议: 买入/不建议
置信: 高/中/低
理由: 一句话"""

        try:
            raw = self._call_llm(prompt)
            log.info(f"[LLM] {symbol} 原始回复: {raw[:100]}")

            # 解析回复
            should_buy = "买入" in raw and "不建议" not in raw
            confidence = "medium"
            if "高" in raw[:20]:
                confidence = "high"
            elif "低" in raw[:20]:
                confidence = "low"

            return Opinion(
                should_buy=should_buy,
                confidence=confidence,
                text=raw,
            )
        except Exception as e:
            log.warning(f"[LLM] {symbol} 调用失败: {e}")
            # 失败时不阻止交易，默认同意
            return Opinion(should_buy=True, confidence="low", text=f"LLM不可用: {e}")

    def check_health(self) -> bool:
        """检查 LLM 服务是否在线"""
        try:
            resp = requests.get(
                self.api_url.rsplit("/", 1)[0].replace("generate", "tags"),
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False
