#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResilientLLMClient — 双 Provider 自动故障切换 LLM 客户端

设计目标：
  - Hunyuan / Qwen 两个 LLM 源交替使用，任一失败自动切换到另一个
  - 不管怎么样一直重试，直到成功或达到全局上限
  - 每次 provider 失败后 sleep 1s 再切换，避免瞬间压垮两个 API
  - 接口与 HunyuanApiClient / QwenChatApiClient 完全兼容（duck-typing）

调用示例：
    from resilient_llm_client import ResilientLLMClient

    client = ResilientLLMClient()
    resp, metrics = client.chat(messages=[...])
    content = resp["choices"][0]["message"]["content"]

切换流程：
    尝试 Provider A → 内置重试 → 全部失败
    → sleep 1s → 切换到 Provider B → 内置重试 → 全部失败
    → sleep 1s → 切换到 Provider A → ...
    → 循环直到成功或 max_provider_switches 轮次耗尽
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("resilient_llm_client")


@dataclass
class RequestMetrics:
    """与 HunyuanApiClient.RequestMetrics 兼容的指标类。"""
    success: bool
    latency_s: float
    attempt: int
    status_code: Optional[int]
    error: Optional[str]
    provider: str = ""


# ─────────────────── Provider 配置 ───────────────────

# Hunyuan 内网 API（默认主选）
HUNYUAN_CONFIG = {
    "api_key": os.environ.get("HUNYUAN_API_KEY", ""),
    "base_url": "http://hunyuanapi.woa.com",
    "model": "hunyuan-2.0-instruct-20251111",
}

# Qwen DashScope API（备选）
QWEN_CONFIG = {
    "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-plus-latest",
}


def _create_hunyuan_client(
    timeout: int = 120,
    max_retries: int = 3,
    **extra_kwargs,
):
    """延迟导入并创建 HunyuanApiClient 实例。"""
    from hunyuan_api import HunyuanApiClient  # type: ignore
    kwargs = {
        "api_key": HUNYUAN_CONFIG["api_key"],
        "base_url": HUNYUAN_CONFIG["base_url"],
        "model": HUNYUAN_CONFIG["model"],
        "timeout": timeout,
        "max_retries": max_retries,
        "retry_base_delay": 1.0,
        "retry_jitter": 0.3,
        "retry_backoff_factor": 2.0,
        "max_retry_delay": 30.0,
    }
    kwargs.update(extra_kwargs)
    client = HunyuanApiClient(**kwargs)
    # 放宽 overload 检测：由 ResilientLLMClient 层面控制切换
    client.overload_threshold = 50     # 几乎不触发
    client.overload_cooldown = 10      # 即使触发也只冷却 10 秒
    return client


def _create_qwen_client(
    timeout: int = 120,
    max_retries: int = 3,
    **extra_kwargs,
):
    """延迟导入并创建 QwenChatApiClient 实例。"""
    from qwen_chat_api import QwenChatApiClient  # type: ignore
    kwargs = {
        "api_key": QWEN_CONFIG["api_key"],
        "base_url": QWEN_CONFIG["base_url"],
        "model": QWEN_CONFIG["model"],
        "timeout": timeout,
        "max_retries": max_retries,
        "retry_base_delay": 1.0,
        "retry_jitter": 0.3,
        "retry_backoff_factor": 2.0,
        "max_retry_delay": 30.0,
    }
    kwargs.update(extra_kwargs)
    client = QwenChatApiClient(**kwargs)
    # 放宽 overload 检测
    client.overload_threshold = 50
    client.overload_cooldown = 10
    return client


class ResilientLLMClient:
    """双 Provider 自动故障切换 LLM 客户端。

    接口与 HunyuanApiClient / QwenChatApiClient 完全兼容。

    Args:
        max_provider_switches: 最多切换多少轮（每轮 = 尝试一个 provider 的完整重试）。
            默认 6 轮，即 Hunyuan→Qwen→Hunyuan→Qwen→Hunyuan→Qwen。
        switch_sleep: 切换 provider 前的 sleep 秒数，默认 1.0。
        timeout: 单次 HTTP 请求超时（秒）。
        max_retries_per_provider: 每个 provider 内部重试次数。
        primary: 首选 provider，"hunyuan" 或 "qwen"。
    """

    def __init__(
        self,
        max_provider_switches: int = 6,
        switch_sleep: float = 1.0,
        timeout: int = 120,
        max_retries_per_provider: int = 3,
        primary: str = "hunyuan",
        hunyuan_kwargs: Optional[Dict[str, Any]] = None,
        qwen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.max_provider_switches = max_provider_switches
        self.switch_sleep = switch_sleep
        self.timeout = timeout
        self.max_retries_per_provider = max_retries_per_provider

        # 延迟创建 client（避免导入时就连接）
        self._hunyuan = None
        self._qwen = None
        self._hunyuan_kwargs = hunyuan_kwargs or {}
        self._qwen_kwargs = qwen_kwargs or {}

        # 统计
        self.total_calls = 0
        self.total_switches = 0
        self.provider_stats = {"hunyuan": {"ok": 0, "fail": 0}, "qwen": {"ok": 0, "fail": 0}}

        # provider 顺序
        if primary == "qwen":
            self._provider_order = ["qwen", "hunyuan"]
        else:
            self._provider_order = ["hunyuan", "qwen"]

        # 兼容属性：部分调用方会读 client.model
        self.model = HUNYUAN_CONFIG["model"] if primary != "qwen" else QWEN_CONFIG["model"]

    def _get_client(self, name: str):
        """获取或延迟创建指定 provider 的 client。"""
        if name == "hunyuan":
            if self._hunyuan is None:
                self._hunyuan = _create_hunyuan_client(
                    timeout=self.timeout,
                    max_retries=self.max_retries_per_provider,
                    **self._hunyuan_kwargs,
                )
                LOGGER.info("ResilientLLM: Hunyuan client created (endpoint=%s, model=%s)",
                            self._hunyuan.endpoint, self._hunyuan.model)
            return self._hunyuan
        else:
            if self._qwen is None:
                self._qwen = _create_qwen_client(
                    timeout=self.timeout,
                    max_retries=self.max_retries_per_provider,
                    **self._qwen_kwargs,
                )
                LOGGER.info("ResilientLLM: Qwen client created (endpoint=%s, model=%s)",
                            self._qwen.endpoint, self._qwen.model)
            return self._qwen

    def chat(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        request_overrides: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Tuple[Dict[str, Any], RequestMetrics]:
        """与 HunyuanApiClient.chat() / QwenChatApiClient.chat() 完全兼容。

        自动在 Hunyuan ↔ Qwen 之间切换，直到成功或穷尽所有轮次。
        """
        self.total_calls += 1
        start_ts = time.monotonic()
        last_error: Optional[Exception] = None
        last_provider = ""

        for switch_round in range(self.max_provider_switches):
            provider_name = self._provider_order[switch_round % len(self._provider_order)]

            # 切换 provider 前 sleep（第一次不 sleep）
            if switch_round > 0:
                LOGGER.info(
                    "ResilientLLM: switching to %s (round %d/%d), sleep %.1fs",
                    provider_name, switch_round + 1, self.max_provider_switches, self.switch_sleep,
                )
                self.total_switches += 1
                time.sleep(self.switch_sleep)

            client = self._get_client(provider_name)
            last_provider = provider_name

            try:
                kwargs = {"messages": messages, "stream": stream, "debug": debug}
                if request_overrides:
                    kwargs["request_overrides"] = request_overrides
                resp, inner_metrics = client.chat(**kwargs)

                # 成功
                latency = time.monotonic() - start_ts
                self.provider_stats[provider_name]["ok"] += 1

                metrics = RequestMetrics(
                    success=True,
                    latency_s=latency,
                    attempt=switch_round + 1,
                    status_code=inner_metrics.status_code if hasattr(inner_metrics, 'status_code') else 200,
                    error=None,
                    provider=provider_name,
                )

                if switch_round > 0:
                    LOGGER.info(
                        "ResilientLLM: success via %s after %d switch(es), latency=%.1fs",
                        provider_name, switch_round, latency,
                    )

                return resp, metrics

            except Exception as e:
                last_error = e
                self.provider_stats[provider_name]["fail"] += 1

                error_str = str(e)[:200]
                status = getattr(e, "status_code", None)
                LOGGER.warning(
                    "ResilientLLM: %s failed (round %d/%d, status=%s): %s",
                    provider_name, switch_round + 1, self.max_provider_switches,
                    status, error_str,
                )

        # 所有轮次都失败
        latency = time.monotonic() - start_ts
        LOGGER.error(
            "ResilientLLM: ALL %d rounds failed (%.1fs total). "
            "stats: hunyuan=%s, qwen=%s. Last error: %s",
            self.max_provider_switches, latency,
            self.provider_stats["hunyuan"], self.provider_stats["qwen"],
            str(last_error)[:300],
        )

        # 兼容：抛出与底层 client 相同类型的异常
        raise last_error  # type: ignore

    def get_stats_summary(self) -> str:
        """返回可读的统计摘要。"""
        return (
            f"ResilientLLM stats: calls={self.total_calls}, switches={self.total_switches}, "
            f"hunyuan={self.provider_stats['hunyuan']}, qwen={self.provider_stats['qwen']}"
        )
