"""LLM 提供商基本介面。"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Set
import time
import logging

from ..models.base import LLMModel

# 設置日誌
logger = logging.getLogger(__name__)


class LLMResponse:
    """標準化的 LLM 回應類別。
    
    這個類代表來自各種LLM提供商的標準化回應格式。
    """
    
    def __init__(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        latency: Optional[float] = None
    ):
        """初始化LLM回應。
        
        Args:
            content: 回應內容
            metadata: 額外的回應元數據
            model_id: 產生回應的模型ID
            usage: Token使用情況，如 {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            latency: 回應延遲（秒）
        """
        self.content = content
        self.metadata = metadata or {}
        self.model_id = model_id
        self.usage = usage or {}
        self.latency = latency
    
    def __str__(self) -> str:
        """返回回應的字串表示。"""
        return self.content
    
    def __repr__(self) -> str:
        """返回回應的程式表示。"""
        return f"LLMResponse(content='{self.content[:50]}...', model='{self.model_id}')"


class LLMProvider(ABC):
    """LLM 提供商介面。
    
    定義所有LLM提供商必須實作的方法。
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """提供商名稱。"""
        pass
    
    @property
    @abstractmethod
    def models(self) -> Dict[str, LLMModel]:
        """此提供商支援的模型映射。"""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """此提供商的預設模型ID。"""
        pass
    
    @property
    def capabilities(self) -> Set[str]:
        """此提供商支援的功能集合。
        
        預設實現是所有模型功能的聯集。
        """
        all_capabilities = set()
        for model in self.models.values():
            all_capabilities.update(model.capabilities)
        return all_capabilities
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0.5, 
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """聊天功能，發送消息並獲取回應。
        
        Args:
            messages: 消息列表，每個消息是包含'role'和'content'的字典
            model: 要使用的模型ID，如果為None則使用預設模型
            temperature: 溫度參數，控制隨機性
            stream: 是否以串流方式返回回應
            **kwargs: 額外的提供商特定參數
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        pass
    
    @abstractmethod
    def embed(
        self, 
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """生成文字嵌入向量。
        
        Args:
            text: 要嵌入的文字或文字列表
            model: 要使用的模型ID，如果為None則使用預設嵌入模型
            **kwargs: 額外的提供商特定參數
            
        Returns:
            List[float] 或 List[List[float]]: 文字的嵌入向量
        """
        pass
    
    @abstractmethod
    def complete(
        self, 
        prompt: str,
        model: Optional[str] = None, 
        temperature: float = 0.5,
        **kwargs
    ) -> LLMResponse:
        """補全文字。
        
        Args:
            prompt: 提示文字
            model: 要使用的模型ID，如果為None則使用預設模型
            temperature: 溫度參數，控制隨機性
            **kwargs: 額外的提供商特定參數
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        pass
    
    def is_supported(self, feature: str) -> bool:
        """檢查提供商是否支援指定功能。
        
        Args:
            feature: 要檢查的功能名稱
            
        Returns:
            bool: 如果提供商支援該功能，則為True，否則為False
        """
        return feature in self.capabilities
    
    def is_model_supported(self, model_id: str) -> bool:
        """檢查提供商是否支援指定模型。
        
        Args:
            model_id: 要檢查的模型ID
            
        Returns:
            bool: 如果提供商支援該模型，則為True，否則為False
        """
        return model_id in self.models
    
    def get_model(self, model_id: Optional[str] = None) -> LLMModel:
        """獲取指定模型的資訊。
        
        Args:
            model_id: 要獲取的模型ID，如果為None則返回預設模型
            
        Returns:
            LLMModel: 模型對象
            
        Raises:
            ValueError: 如果模型不支援
        """
        model_id = model_id or self.default_model
        if not self.is_model_supported(model_id):
            raise ValueError(f"Model '{model_id}' is not supported by {self.name} provider")
        return self.models[model_id]
    
    def _timed_execution(self, func, *args, **kwargs):
        """執行函數並記錄執行時間。
        
        Args:
            func: 要執行的函數
            *args: 傳遞給函數的位置參數
            **kwargs: 傳遞給函數的關鍵字參數
            
        Returns:
            tuple: (函數返回值, 執行時間(秒))
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result, time.time() - start_time
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise