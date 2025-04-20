"""LLM 使用指標收集服務。"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, DefaultDict
from collections import defaultdict

# 設置日誌
logger = logging.getLogger(__name__)


class LLMMetrics:
    """LLM 使用指標收集服務。
    
    用於收集和報告LLM使用情況的指標，如API調用次數、token使用量、延遲等。
    """
    
    def __init__(self):
        """初始化LLM指標收集器。"""
        # 請求計數，按提供商、模型分組
        self._request_count: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Token使用量，按提供商、模型分組
        self._token_usage: DefaultDict[str, DefaultDict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
        )
        
        # 延遲統計，按提供商、模型分組
        self._latency: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # 錯誤計數，按提供商、錯誤類型分組
        self._errors: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # 成本估算，按提供商分組
        self._cost_estimate: DefaultDict[str, float] = defaultdict(float)
        
        # 時間系列數據，用於繪製趨勢
        self._time_series: List[Dict[str, Any]] = []
        
        # 啟動時間
        self._start_time = time.time()
    
    def record_request(
        self, 
        provider: str, 
        model: str, 
        tokens: Optional[Dict[str, int]] = None,
        operation: str = "chat"
    ) -> None:
        """記錄一個請求。
        
        Args:
            provider: 提供商名稱
            model: 模型ID
            tokens: Token使用情況，例如 {"prompt": 10, "completion": 0, "total": 10}
            operation: 操作類型，例如 "chat", "embed", "complete"
        """
        # 增加請求計數
        self._request_count[provider][model] += 1
        
        # 記錄token使用（如果提供）
        if tokens:
            for key, value in tokens.items():
                self._token_usage[provider][model][key] += value
        
        # 添加時間系列資料點
        self._time_series.append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "operation": operation,
            "tokens": tokens
        })
        
        logger.debug(f"Recorded {operation} request for {provider}/{model}")
    
    def record_response(
        self, 
        provider: str, 
        model: str, 
        tokens: Optional[Dict[str, int]] = None,
        latency: Optional[float] = None,
        operation: str = "chat",
        cost: Optional[float] = None
    ) -> None:
        """記錄一個回應。
        
        Args:
            provider: 提供商名稱
            model: 模型ID
            tokens: Token使用情況，例如 {"prompt": 10, "completion": 20, "total": 30}
            latency: 回應延遲（秒）
            operation: 操作類型，例如 "chat", "embed", "complete"
            cost: 估計成本（美元）
        """
        # 記錄token使用（如果提供）
        if tokens:
            for key, value in tokens.items():
                self._token_usage[provider][model][key] += value
        
        # 記錄延遲（如果提供）
        if latency is not None:
            self._latency[provider][model].append(latency)
        
        # 記錄成本（如果提供）
        if cost is not None:
            self._cost_estimate[provider] += cost
        
        # 更新時間系列資料點
        self._time_series.append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "operation": operation,
            "tokens": tokens,
            "latency": latency,
            "cost": cost
        })
        
        logger.debug(f"Recorded {operation} response for {provider}/{model}")
    
    def record_error(self, provider: str, error_type: str) -> None:
        """記錄一個錯誤。
        
        Args:
            provider: 提供商名稱
            error_type: 錯誤類型
        """
        self._errors[provider][error_type] += 1
        
        # 添加時間系列資料點
        self._time_series.append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "error": error_type
        })
        
        logger.debug(f"Recorded error of type {error_type} for {provider}")
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取彙總的指標統計資料。
        
        Returns:
            Dict[str, Any]: 包含彙總指標的字典
        """
        # 計算總請求數
        total_requests = sum(
            sum(counts.values()) for counts in self._request_count.values()
        )
        
        # 計算總token使用量
        total_tokens = sum(
            sum(usage.get("total", 0) for usage in model_usage.values())
            for model_usage in self._token_usage.values()
        )
        
        # 計算平均延遲
        all_latencies = [
            latency
            for provider_latencies in self._latency.values()
            for model_latencies in provider_latencies.values()
            for latency in model_latencies
        ]
        
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        
        # 計算總錯誤數
        total_errors = sum(
            sum(counts.values()) for counts in self._errors.values()
        )
        
        # 計算總成本
        total_cost = sum(self._cost_estimate.values())
        
        # 計算運行時間
        uptime = time.time() - self._start_time
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "avg_latency": avg_latency,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests) if total_requests > 0 else 0,
            "total_cost": total_cost,
            "uptime": uptime,
            "requests_per_minute": (total_requests / (uptime / 60)) if uptime > 0 else 0,
        }
    
    def get_provider_stats(self, provider: str) -> Dict[str, Any]:
        """獲取特定提供商的指標統計資料。
        
        Args:
            provider: 提供商名稱
            
        Returns:
            Dict[str, Any]: 包含提供商指標的字典
        """
        # 計算提供商的總請求數
        provider_requests = sum(self._request_count[provider].values())
        
        # 計算提供商的總token使用量
        provider_tokens = sum(
            usage.get("total", 0) for usage in self._token_usage[provider].values()
        )
        
        # 計算提供商的平均延遲
        provider_latencies = [
            latency
            for model_latencies in self._latency[provider].values()
            for latency in model_latencies
        ]
        
        provider_avg_latency = (
            sum(provider_latencies) / len(provider_latencies) if provider_latencies else 0
        )
        
        # 計算提供商的總錯誤數
        provider_errors = sum(self._errors[provider].values())
        
        # 計算提供商的總成本
        provider_cost = self._cost_estimate[provider]
        
        return {
            "requests": provider_requests,
            "tokens": provider_tokens,
            "avg_latency": provider_avg_latency,
            "errors": provider_errors,
            "error_rate": (provider_errors / provider_requests) if provider_requests > 0 else 0,
            "cost": provider_cost,
        }
    
    def get_model_stats(self, provider: str, model: str) -> Dict[str, Any]:
        """獲取特定模型的指標統計資料。
        
        Args:
            provider: 提供商名稱
            model: 模型ID
            
        Returns:
            Dict[str, Any]: 包含模型指標的字典
        """
        # 獲取模型的請求數
        model_requests = self._request_count[provider][model]
        
        # 獲取模型的token使用量
        model_tokens = self._token_usage[provider][model].get("total", 0)
        
        # 計算模型的平均延遲
        model_latencies = self._latency[provider][model]
        model_avg_latency = sum(model_latencies) / len(model_latencies) if model_latencies else 0
        
        return {
            "requests": model_requests,
            "tokens": model_tokens,
            "prompt_tokens": self._token_usage[provider][model].get("prompt", 0),
            "completion_tokens": self._token_usage[provider][model].get("completion", 0),
            "avg_latency": model_avg_latency,
        }
    
    def reset(self) -> None:
        """重置所有指標。"""
        self._request_count.clear()
        self._token_usage.clear()
        self._latency.clear()
        self._errors.clear()
        self._cost_estimate.clear()
        self._time_series.clear()
        self._start_time = time.time()
        
        logger.debug("Metrics reset")
    
    def get_time_series(self) -> List[Dict[str, Any]]:
        """獲取時間系列數據。
        
        Returns:
            List[Dict[str, Any]]: 時間系列資料點列表
        """
        return self._time_series