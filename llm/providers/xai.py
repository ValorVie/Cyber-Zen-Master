"""X.AI Grok LLM 提供商實現。"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Set

from openai import OpenAI

from .base import LLMProvider, LLMResponse
from ..models.base import LLMModel

# 設置日誌
logger = logging.getLogger(__name__)


class XAIModel(LLMModel):
    """X.AI Grok模型實現。"""
    
    def __init__(
        self, 
        id: str, 
        context_length: int,
        capabilities: Set[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """初始化X.AI模型。
        
        Args:
            id: 模型ID
            context_length: 模型支援的最大上下文長度（以token為單位）
            capabilities: 模型支援的功能集合
            name: 模型顯示名稱
            description: 模型描述
        """
        self._id = id
        self._context_length = context_length
        self._capabilities = capabilities
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


class XAIProvider(LLMProvider):
    """X.AI LLM提供商實現。
    
    支援X.AI的Grok模型，通過OpenAI兼容的API訪問。
    """
    
    def __init__(
        self, 
        api_key: str,
        base_url: str = "https://api.x.ai/v1"
    ):
        """初始化X.AI提供商。
        
        Args:
            api_key: X.AI API金鑰
            base_url: API基礎URL
        """
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 定義支援的模型
        self._models = {
            "grok-3-beta": XAIModel(
                id="grok-3-beta",
                name="Grok-3 Beta",
                context_length=128000,
                capabilities={"chat", "completion"},
                description="X.AI的Grok-3模型，具有強大的對話和補全能力"
            ),
        }
        
        self._default_model = "grok-3-beta"
        
        logger.info("Initialized X.AI provider")
    
    @property
    def name(self) -> str:
        """獲取提供商名稱。"""
        return "xai"
    
    @property
    def models(self) -> Dict[str, LLMModel]:
        """獲取支援的模型映射。"""
        return self._models
    
    @property
    def default_model(self) -> str:
        """獲取預設模型ID。"""
        return self._default_model
    
    def set_default_model(self, model_id: str) -> None:
        """設置預設模型ID。
        
        Args:
            model_id: 模型ID
            
        Raises:
            ValueError: 如果模型不支援
        """
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' is not supported by X.AI provider")
        
        self._default_model = model_id
        logger.info(f"Set default model to: {model_id}")
    
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
            **kwargs: 額外的參數，如max_tokens, top_p, stop等
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        model_id = model or self.default_model
        model_obj = self.get_model(model_id)  # 驗證模型是否支援
        
        if "chat" not in model_obj.capabilities:
            raise ValueError(f"Model '{model_id}' does not support chat capability")
        
        logger.debug(f"Chat request to {model_id} with {len(messages)} messages")
        
        start_time = time.time()
        try:
            response = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
            
            if stream:
                # 處理流式回應
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        if kwargs.get("verbose", False):
                            print(content, end='', flush=True)
                        full_response += content
                
                if kwargs.get("verbose", False):
                    print()
                
                latency = time.time() - start_time
                return LLMResponse(
                    content=full_response,
                    model_id=model_id,
                    latency=latency
                )
            else:
                # 處理非流式回應
                latency = time.time() - start_time
                usage = None
                if hasattr(response, "usage"):
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    model_id=model_id,
                    usage=usage,
                    latency=latency,
                    metadata={"finish_reason": response.choices[0].finish_reason} if hasattr(response.choices[0], "finish_reason") else {}
                )
                
        except Exception as e:
            logger.error(f"X.AI chat error: {str(e)}")
            raise
    
    def embed(
        self, 
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """生成文字嵌入向量。
        
        Args:
            text: 要嵌入的文字或文字列表
            model: 要使用的模型ID
            **kwargs: 額外的參數
            
        Raises:
            NotImplementedError: X.AI API尚未提供嵌入功能
        """
        # 目前X.AI API尚未提供嵌入功能
        raise NotImplementedError("X.AI provider does not support embeddings yet")
    
    def complete(
        self, 
        prompt: str,
        model: Optional[str] = None, 
        temperature: float = 0.5,
        **kwargs
    ) -> LLMResponse:
        """補全文字，使用聊天API實現。
        
        Args:
            prompt: 提示文字
            model: 要使用的模型ID，如果為None則使用預設模型
            temperature: 溫度參數，控制隨機性
            **kwargs: 額外的參數，如max_tokens, top_p, stop等
            
        Returns:
            LLMResponse: 標準化的回應對象
        """
        # 使用聊天API實現補全功能
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            **kwargs
        )