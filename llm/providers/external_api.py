"""外部 API 提供商實現。

這個模組提供了一個靈活的提供商，用於整合各種外部 API 服務。
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Set
import requests

from .base import LLMProvider, LLMResponse
from ..models.base import LLMModel

# 設置日誌
logger = logging.getLogger(__name__)


class APIEndpoint:
    """外部API端點的配置和處理。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化API端點配置。
        
        Args:
            config: 端點配置字典，至少應包含'url'
        """
        self.url = config['url']
        self.method = config.get('method', 'POST').upper()
        self.headers = config.get('headers', {})
        self.response_path = config.get('response_path', None)
        self.timeout = config.get('timeout', 30)
        self.retry_count = config.get('retry_count', 3)
        self.retry_delay = config.get('retry_delay', 1)
    
    def call(self, params: Dict[str, Any]) -> Any:
        """呼叫API端點並返回回應。
        
        Args:
            params: 呼叫參數
            
        Returns:
            Any: API回應
            
        Raises:
            Exception: 如果API呼叫失敗
        """
        headers = {**self.headers, 'Content-Type': 'application/json'}
        data = json.dumps(params)
        
        for attempt in range(self.retry_count):
            try:
                if self.method == 'GET':
                    response = requests.get(
                        self.url, 
                        params=params, 
                        headers=headers, 
                        timeout=self.timeout
                    )
                else:  # POST, PUT, DELETE等
                    response = requests.request(
                        method=self.method,
                        url=self.url,
                        data=data,
                        headers=headers,
                        timeout=self.timeout
                    )
                
                response.raise_for_status()
                result = response.json()
                
                # 如果指定了回應路徑，提取特定部分
                if self.response_path:
                    parts = self.response_path.split('.')
                    for part in parts:
                        if isinstance(result, dict) and part in result:
                            result = result[part]
                        else:
                            break
                
                return result
            
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt == self.retry_count - 1:  # 最後一次嘗試
                    raise Exception(f"API呼叫失敗: {str(e)}")
                time.sleep(self.retry_delay)
        
        # 不應該達到這裡，但為了安全
        raise Exception("API呼叫重試後仍然失敗")


class ResponseTransformer:
    """API回應轉換器，將外部API回應轉換為標準LLM回應格式。"""
    
    def transform_to_chat_response(self, raw_response: Any) -> LLMResponse:
        """將原始回應轉換為聊天回應格式。
        
        Args:
            raw_response: 原始API回應
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        if isinstance(raw_response, dict):
            # 嘗試提取內容，實際實現會根據API回應結構調整
            content = raw_response.get('text') or raw_response.get('content') or str(raw_response)
            return LLMResponse(content=content, metadata=raw_response)
        elif isinstance(raw_response, str):
            return LLMResponse(content=raw_response)
        else:
            return LLMResponse(content=str(raw_response), metadata={"raw": raw_response})
    
    def transform_to_embedding(self, raw_response: Any) -> List[float]:
        """將原始回應轉換為嵌入向量格式。
        
        Args:
            raw_response: 原始API回應
            
        Returns:
            List[float]: 嵌入向量
            
        Raises:
            ValueError: 如果無法從回應中提取嵌入向量
        """
        if isinstance(raw_response, dict) and 'embedding' in raw_response:
            return raw_response['embedding']
        elif isinstance(raw_response, list) and all(isinstance(x, (int, float)) for x in raw_response):
            return raw_response
        else:
            raise ValueError(f"無法從回應中提取嵌入向量: {raw_response}")


class ExternalAPIModel(LLMModel):
    """外部API模型實現。"""
    
    def __init__(
        self, 
        id: str, 
        capabilities: Set[str],
        context_length: int = 0,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """初始化外部API模型。
        
        Args:
            id: 模型ID
            capabilities: 模型支援的功能集合
            context_length: 模型支援的最大上下文長度，對於外部API通常不適用
            name: 模型顯示名稱
            description: 模型描述
        """
        self._id = id
        self._capabilities = capabilities
        self._context_length = context_length
        self._name = name or id
        self._description = description
    
    @property
    def id(self) -> str:
        """獲取模型ID。"""
        return self._id
    
    @property
    def context_length(self) -> int:
        """獲取最大上下文長度。"""
        return self._context_length
    
    @property
    def capabilities(self) -> Set[str]:
        """獲取支援的功能集合。"""
        return self._capabilities
    
    @property
    def name(self) -> str:
        """獲取模型顯示名稱。"""
        return self._name
    
    @property
    def description(self) -> Optional[str]:
        """獲取模型描述。"""
        return self._description


class ExternalAPIProvider(LLMProvider):
    """外部API提供商實現。
    
    允許整合各種外部API服務，並將它們轉換為統一的LLM介面。
    """
    
    def __init__(self, endpoints: Optional[Dict[str, Dict[str, Any]]] = None):
        """初始化外部API提供商。
        
        Args:
            endpoints: 初始端點配置字典，格式為 {端點名稱: 配置字典}
        """
        self._endpoints: Dict[str, APIEndpoint] = {}
        self._transformer = ResponseTransformer()
        
        # 註冊初始端點
        if endpoints:
            for name, config in endpoints.items():
                self.register_endpoint(name, config)
        
        # 定義一個虛擬模型用於API呼叫
        self._models = {
            "external-api": ExternalAPIModel(
                id="external-api",
                name="External API",
                capabilities={"chat", "completion", "embedding", "external_api"},
                description="Interface for calling external APIs"
            )
        }
        
        self._default_model = "external-api"
        
        logger.info("Initialized External API provider")
    
    @property
    def name(self) -> str:
        """獲取提供商名稱。"""
        return "external"
    
    @property
    def models(self) -> Dict[str, LLMModel]:
        """獲取支援的模型映射。"""
        return self._models
    
    @property
    def default_model(self) -> str:
        """獲取預設模型ID。"""
        return self._default_model
    
    @property
    def endpoints(self) -> List[str]:
        """獲取所有註冊的端點名稱。
        
        Returns:
            List[str]: 端點名稱列表
        """
        return list(self._endpoints.keys())
    
    def register_endpoint(self, name: str, config: Dict[str, Any]) -> None:
        """註冊新的API端點。
        
        Args:
            name: 端點名稱
            config: 端點配置字典
        """
        self._endpoints[name] = APIEndpoint(config)
        logger.info(f"Registered endpoint: {name}")
    
    def set_response_transformer(self, transformer: ResponseTransformer) -> None:
        """設置自定義的回應轉換器。
        
        Args:
            transformer: 新的回應轉換器實例
        """
        self._transformer = transformer
        logger.info("Set custom response transformer")
    
    def call_endpoint(self, name: str, params: Dict[str, Any]) -> Any:
        """呼叫指定的API端點。
        
        Args:
            name: 端點名稱
            params: 呼叫參數
            
        Returns:
            Any: API回應
            
        Raises:
            ValueError: 如果端點不存在
        """
        if name not in self._endpoints:
            raise ValueError(f"未知的API端點: {name}")
        
        logger.debug(f"Calling endpoint: {name}")
        start_time = time.time()
        
        try:
            result = self._endpoints[name].call(params)
            latency = time.time() - start_time
            logger.debug(f"API call completed in {latency:.2f}s")
            return result
        
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            raise
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0.5, 
        stream: bool = False,
        endpoint: str = "chat",
        **kwargs
    ) -> LLMResponse:
        """聊天功能，映射到指定的API端點。
        
        Args:
            messages: 消息列表，每個消息是包含'role'和'content'的字典
            model: 要使用的模型ID，對於外部API通常忽略
            temperature: 溫度參數，控制隨機性
            stream: 是否以串流方式返回回應
            endpoint: 要使用的API端點名稱
            **kwargs: 額外的參數
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        if endpoint not in self._endpoints:
            raise ValueError(f"未知的API端點: {endpoint}")
        
        # 構建API請求參數
        params = {
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        # 呼叫API端點
        start_time = time.time()
        raw_response = self.call_endpoint(endpoint, params)
        latency = time.time() - start_time
        
        # 轉換回應格式
        response = self._transformer.transform_to_chat_response(raw_response)
        response.latency = latency
        response.model_id = model or self.default_model
        
        return response
    
    def embed(
        self, 
        text: Union[str, List[str]],
        model: Optional[str] = None,
        endpoint: str = "embedding",
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """嵌入功能，映射到指定的API端點。
        
        Args:
            text: 要嵌入的文字或文字列表
            model: 要使用的模型ID，對於外部API通常忽略
            endpoint: 要使用的API端點名稱
            **kwargs: 額外的參數
            
        Returns:
            Union[List[float], List[List[float]]]: 文字的嵌入向量
        """
        if endpoint not in self._endpoints:
            raise ValueError(f"未知的API端點: {endpoint}")
        
        # 確保文字以列表形式提供
        input_texts = [text] if isinstance(text, str) else text
        
        # 構建API請求參數
        params = {
            "text": input_texts,
            **kwargs
        }
        
        # 呼叫API端點
        raw_response = self.call_endpoint(endpoint, params)
        
        # 轉換回應格式
        embeddings = self._transformer.transform_to_embedding(raw_response)
        
        # 如果只嵌入一個文字，返回單個向量
        if isinstance(text, str) and isinstance(embeddings, list) and all(isinstance(x, list) for x in embeddings):
            return embeddings[0]
        
        return embeddings
    
    def complete(
        self, 
        prompt: str,
        model: Optional[str] = None, 
        temperature: float = 0.5,
        endpoint: str = "complete",
        **kwargs
    ) -> LLMResponse:
        """補全功能，映射到指定的API端點。
        
        Args:
            prompt: 提示文字
            model: 要使用的模型ID，對於外部API通常忽略
            temperature: 溫度參數，控制隨機性
            endpoint: 要使用的API端點名稱
            **kwargs: 額外的參數
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        if endpoint not in self._endpoints:
            raise ValueError(f"未知的API端點: {endpoint}")
        
        # 構建API請求參數
        params = {
            "prompt": prompt,
            "temperature": temperature,
            **kwargs
        }
        
        # 呼叫API端點
        start_time = time.time()
        raw_response = self.call_endpoint(endpoint, params)
        latency = time.time() - start_time
        
        # 轉換回應格式
        response = self._transformer.transform_to_chat_response(raw_response)
        response.latency = latency
        response.model_id = model or self.default_model
        
        return response
    
    def process_multimodal_content(
        self, 
        content: Union[str, Dict, List],
        endpoint: str = "multimodal",
        **kwargs
    ) -> LLMResponse:
        """處理多模態內容。
        
        Args:
            content: 要處理的多模態內容
            endpoint: 要使用的API端點名稱
            **kwargs: 額外的參數
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        if endpoint not in self._endpoints:
            raise ValueError(f"未知的API端點: {endpoint}")
        
        # 構建適合多模態API的請求格式
        if isinstance(content, str):
            params = {"text": content, **kwargs}
        elif isinstance(content, dict):
            params = {**content, **kwargs}
        else:
            params = {"content": content, **kwargs}
        
        # 呼叫API端點
        start_time = time.time()
        raw_response = self.call_endpoint(endpoint, params)
        latency = time.time() - start_time
        
        # 轉換回應格式
        response = self._transformer.transform_to_chat_response(raw_response)
        response.latency = latency
        
        return response