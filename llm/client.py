"""LLM客戶端，整合和管理多個提供商。"""

import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

from .providers.base import LLMProvider, LLMResponse
from .services.cache import LLMCache
from .services.metrics import LLMMetrics

# 設置日誌
logger = logging.getLogger(__name__)


class LLMClient:
    """LLM客戶端，統一介面用於整合多個提供商。
    
    這個類是整個架構的中心協調器，負責管理不同的提供商、服務和策略。
    """
    
    def __init__(
        self,
        providers: Optional[Dict[str, LLMProvider]] = None,
        default_provider: Optional[str] = None,
        cache: Optional[LLMCache] = None,
        metrics: Optional[LLMMetrics] = None,
        retry_attempts: int = 3,
        enable_fallback: bool = True
    ):
        """初始化LLM客戶端。
        
        Args:
            providers: 提供商映射，鍵為提供商名稱，值為LLMProvider實例
            default_provider: 預設提供商名稱
            cache: LLMCache實例，如果為None則創建新實例
            metrics: LLMMetrics實例，如果為None則創建新實例
            retry_attempts: 請求失敗時的重試次數
            enable_fallback: 是否啟用提供商故障轉移
        """
        self._providers = providers or {}
        self._default_provider = default_provider
        self._cache = cache or LLMCache()
        self._metrics = metrics or LLMMetrics()
        self._retry_attempts = retry_attempts
        self._enable_fallback = enable_fallback
        
        # 用於每種操作的鏈式處理器
        self._pre_processors: Dict[str, List[Callable]] = {
            "chat": [],
            "embed": [],
            "complete": [],
            "call": []
        }
        self._post_processors: Dict[str, List[Callable]] = {
            "chat": [],
            "embed": [],
            "complete": [],
            "call": []
        }
    
    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """註冊一個LLM提供商。
        
        Args:
            name: 提供商名稱
            provider: LLMProvider實例
        """
        self._providers[name] = provider
        if self._default_provider is None:
            self._default_provider = name
        
        logger.info(f"Registered provider: {name}")
    
    def set_default_provider(self, name: str) -> None:
        """設置預設提供商。
        
        Args:
            name: 提供商名稱
            
        Raises:
            ValueError: 如果提供商未註冊
        """
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' is not registered")
        
        self._default_provider = name
        logger.info(f"Set default provider to: {name}")
    
    def get_provider(self, name: Optional[str] = None) -> LLMProvider:
        """獲取指定的提供商。
        
        Args:
            name: 提供商名稱，如果為None則使用預設提供商
            
        Returns:
            LLMProvider: 提供商實例
            
        Raises:
            ValueError: 如果提供商未註冊或未設置預設提供商
        """
        provider_name = name or self._default_provider
        
        if provider_name is None:
            raise ValueError("No default provider set")
        
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' is not registered")
        
        return self._providers[provider_name]
    
    def list_providers(self) -> List[str]:
        """列出所有註冊的提供商。
        
        Returns:
            List[str]: 提供商名稱列表
        """
        return list(self._providers.keys())
    
    def add_pre_processor(self, operation: str, processor: Callable) -> None:
        """添加前置處理器。
        
        Args:
            operation: 操作類型，如"chat", "embed", "complete", "call"
            processor: 處理器函數
        """
        if operation in self._pre_processors:
            self._pre_processors[operation].append(processor)
    
    def add_post_processor(self, operation: str, processor: Callable) -> None:
        """添加後置處理器。
        
        Args:
            operation: 操作類型，如"chat", "embed", "complete", "call"
            processor: 處理器函數
        """
        if operation in self._post_processors:
            self._post_processors[operation].append(processor)
    
    def _apply_processors(self, processors: List[Callable], data: Any) -> Any:
        """應用處理器鏈。
        
        Args:
            processors: 處理器函數列表
            data: 要處理的數據
            
        Returns:
            Any: 處理後的數據
        """
        result = data
        for processor in processors:
            result = processor(result)
        return result
    
    def _get_fallback_provider(self, preferred: str, required_feature: Optional[str] = None) -> Optional[str]:
        """獲取備用提供商。
        
        Args:
            preferred: 首選提供商名稱
            required_feature: 必需的功能
            
        Returns:
            Optional[str]: 備用提供商名稱，如果沒有合適的則為None
        """
        # 篩選掉首選提供商
        candidates = [p for p in self._providers.keys() if p != preferred]
        
        # 如果指定了功能要求，篩選支援該功能的提供商
        if required_feature:
            candidates = [
                p for p in candidates 
                if self._providers[p].is_supported(required_feature)
            ]
        
        # 如果沒有合適的候選項，返回None
        if not candidates:
            return None
        
        # 隨機選擇一個備用提供商
        return random.choice(candidates)
    
    def _execute_with_retry(
        self, 
        operation: str,
        provider: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """執行操作，支援重試和故障轉移。
        
        Args:
            operation: 操作類型
            provider: 提供商名稱
            func: 要執行的函數
            *args: 函數的位置參數
            **kwargs: 函數的關鍵字參數
            
        Returns:
            Any: 函數的返回值
            
        Raises:
            Exception: 如果所有嘗試都失敗
        """
        # 先檢查快取
        cache_key = {
            "operation": operation,
            "provider": provider,
            "args": args,
            "kwargs": kwargs
        }
        
        if operation in ("chat", "embed", "complete") and kwargs.get("use_cache", True):
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # 應用前置處理器
        processed_args = args
        processed_kwargs = kwargs.copy()
        
        # 當有前置處理器時，應用它們
        if operation in self._pre_processors and self._pre_processors[operation]:
            processor_input = {"args": args, "kwargs": kwargs}
            processor_output = self._apply_processors(self._pre_processors[operation], processor_input)
            processed_args = processor_output["args"]
            processed_kwargs = processor_output["kwargs"]
        
        # 保存原始提供商名稱，以備故障轉移
        original_provider = provider
        
        # 清除非操作特定的參數
        operation_kwargs = processed_kwargs.copy()
        if "use_cache" in operation_kwargs:
            del operation_kwargs["use_cache"]
        
        last_error = None
        for attempt in range(self._retry_attempts):
            try:
                provider_instance = self.get_provider(provider)
                
                # 記錄請求指標
                if hasattr(provider_instance, "default_model"):
                    model = operation_kwargs.get("model") or provider_instance.default_model
                    self._metrics.record_request(provider, model, operation=operation)
                
                # 執行操作
                result = func(*processed_args, **operation_kwargs)
                
                # 記錄回應指標
                if isinstance(result, LLMResponse) and hasattr(provider_instance, "default_model"):
                    model = operation_kwargs.get("model") or provider_instance.default_model
                    self._metrics.record_response(
                        provider, 
                        model, 
                        tokens=result.usage if hasattr(result, "usage") else None,
                        latency=result.latency if hasattr(result, "latency") else None,
                        operation=operation
                    )
                
                # 應用後置處理器
                if operation in self._post_processors and self._post_processors[operation]:
                    result = self._apply_processors(self._post_processors[operation], result)
                
                # 快取結果
                if operation in ("chat", "embed", "complete") and kwargs.get("use_cache", True):
                    self._cache.set(cache_key, result)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1} failed for {operation} with {provider}: {str(e)}")
                
                # 記錄錯誤
                self._metrics.record_error(provider, str(type(e).__name__))
                
                # 如果啟用了故障轉移且不是最後一次嘗試，嘗試切換提供商
                if self._enable_fallback and attempt < self._retry_attempts - 1:
                    fallback = self._get_fallback_provider(provider, operation)
                    if fallback:
                        provider = fallback
                        logger.info(f"Switching to fallback provider: {fallback}")
        
        # 如果所有嘗試都失敗，拋出最後一個錯誤
        if last_error:
            logger.error(f"All attempts failed for {operation} with {original_provider}")
            raise last_error
        
        # 這不應該被執行到，但為了安全
        raise Exception(f"Unknown error during {operation} with {original_provider}")
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        provider: Optional[str] = None, 
        model: Optional[str] = None,
        temperature: float = 0.5, 
        stream: bool = False,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """聊天功能，發送消息並獲取回應。
        
        Args:
            messages: 消息列表，每個消息是包含'role'和'content'的字典
            provider: 提供商名稱，如果為None則使用預設提供商
            model: 要使用的模型ID，如果為None則使用提供商的預設模型
            temperature: 溫度參數，控制隨機性
            stream: 是否以串流方式返回回應
            use_cache: 是否使用快取，如果為True則嘗試從快取中獲取回應
            **kwargs: 額外的提供商特定參數
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        provider_name = provider or self._default_provider
        if not provider_name:
            raise ValueError("No provider specified and no default provider set")
        
        def execute_chat():
            provider_instance = self.get_provider(provider_name)
            return provider_instance.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
        
        return self._execute_with_retry(
            operation="chat",
            provider=provider_name,
            func=execute_chat,
            messages=messages,
            model=model,
            temperature=temperature,
            stream=stream,
            use_cache=use_cache,
            **kwargs
        )
    
    def embed(
        self, 
        text: Union[str, List[str]],
        provider: Optional[str] = None, 
        model: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """生成文字嵌入向量。
        
        Args:
            text: 要嵌入的文字或文字列表
            provider: 提供商名稱，如果為None則使用預設提供商
            model: 要使用的模型ID，如果為None則使用提供商的預設嵌入模型
            use_cache: 是否使用快取，如果為True則嘗試從快取中獲取嵌入
            **kwargs: 額外的提供商特定參數
            
        Returns:
            List[float] 或 List[List[float]]: 文字的嵌入向量
        """
        provider_name = provider or self._default_provider
        if not provider_name:
            raise ValueError("No provider specified and no default provider set")
        
        def execute_embed():
            provider_instance = self.get_provider(provider_name)
            return provider_instance.embed(
                text=text,
                model=model,
                **kwargs
            )
        
        return self._execute_with_retry(
            operation="embed",
            provider=provider_name,
            func=execute_embed,
            text=text,
            model=model,
            use_cache=use_cache,
            **kwargs
        )
    
    def complete(
        self, 
        prompt: str,
        provider: Optional[str] = None, 
        model: Optional[str] = None,
        temperature: float = 0.5,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """補全文字。
        
        Args:
            prompt: 提示文字
            provider: 提供商名稱，如果為None則使用預設提供商
            model: 要使用的模型ID，如果為None則使用提供商的預設模型
            temperature: 溫度參數，控制隨機性
            use_cache: 是否使用快取，如果為True則嘗試從快取中獲取回應
            **kwargs: 額外的提供商特定參數
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        provider_name = provider or self._default_provider
        if not provider_name:
            raise ValueError("No provider specified and no default provider set")
        
        def execute_complete():
            provider_instance = self.get_provider(provider_name)
            return provider_instance.complete(
                prompt=prompt,
                model=model,
                temperature=temperature,
                **kwargs
            )
        
        return self._execute_with_retry(
            operation="complete",
            provider=provider_name,
            func=execute_complete,
            prompt=prompt,
            model=model,
            temperature=temperature,
            use_cache=use_cache,
            **kwargs
        )
    
    def call(
        self, 
        provider: str, 
        endpoint: str = None,
        params: Dict[str, Any] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Any:
        """呼叫提供商的特定端點或方法。
        
        這個方法主要用於訪問非標準LLM功能，如外部API。
        
        Args:
            provider: 提供商名稱
            endpoint: 端點或方法名稱
            params: 呼叫參數
            use_cache: 是否使用快取
            **kwargs: 額外參數
            
        Returns:
            Any: 呼叫結果
        """
        provider_instance = self.get_provider(provider)
        
        # 檢查提供商是否支援call_endpoint方法（主要是ExternalAPIProvider）
        if hasattr(provider_instance, "call_endpoint") and callable(getattr(provider_instance, "call_endpoint")):
            def execute_call():
                return provider_instance.call_endpoint(endpoint, params or {})
            
            return self._execute_with_retry(
                operation="call",
                provider=provider,
                func=execute_call,
                endpoint=endpoint,
                params=params,
                use_cache=use_cache,
                **kwargs
            )
        else:
            raise NotImplementedError(f"Provider {provider} does not support the 'call' operation")
    
    @property
    def cache(self) -> LLMCache:
        """獲取快取服務實例。"""
        return self._cache
    
    @property
    def metrics(self) -> LLMMetrics:
        """獲取指標服務實例。"""
        return self._metrics
    
    def get_capabilities(self) -> Dict[str, Set[str]]:
        """獲取所有註冊提供商的功能。
        
        Returns:
            Dict[str, Set[str]]: 提供商名稱到功能集合的映射
        """
        capabilities = {}
        for name, provider in self._providers.items():
            capabilities[name] = provider.capabilities
        return capabilities
    
    def get_models(self, provider: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """獲取提供商支援的模型。
        
        Args:
            provider: 提供商名稱，如果為None則獲取所有提供商的模型
            
        Returns:
            Dict[str, Dict[str, Any]]: 模型ID到模型資訊的映射
        """
        if provider:
            provider_instance = self.get_provider(provider)
            return {model_id: model.__dict__ for model_id, model in provider_instance.models.items()}
        else:
            all_models = {}
            for provider_name, provider_instance in self._providers.items():
                if hasattr(provider_instance, "models"):
                    provider_models = {f"{provider_name}/{model_id}": model.__dict__ 
                                       for model_id, model in provider_instance.models.items()}
                    all_models.update(provider_models)
            return all_models