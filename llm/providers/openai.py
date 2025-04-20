"""OpenAI LLM 提供商實現。"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Set

from openai import OpenAI

from .base import LLMProvider, LLMResponse
from ..models.base import LLMModel

# 設置日誌
logger = logging.getLogger(__name__)


class OpenAIModel(LLMModel):
    """OpenAI模型實現。"""
    
    def __init__(
        self, 
        id: str, 
        context_length: int,
        capabilities: Set[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """初始化OpenAI模型。
        
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


class OpenAIProvider(LLMProvider):
    """OpenAI LLM提供商實現。
    
    支援OpenAI原生API，包括聊天、嵌入和補全功能。
    """
    
    def __init__(
        self, 
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None
    ):
        """初始化OpenAI提供商。
        
        Args:
            api_key: OpenAI API金鑰
            base_url: API基礎URL，預設為OpenAI官方API
            organization: 組織ID（可選）
        """
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization
        )
        
        # 定義支援的模型
        self._models = {
            # GPT-3.5
            "gpt-3.5-turbo": OpenAIModel(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                context_length=16385,
                capabilities={"chat", "completion"},
                description="Optimized for chat, capable of understanding and generating natural language or code"
            ),
            
            # GPT-4
            "gpt-4-turbo": OpenAIModel(
                id="gpt-4-turbo",
                name="GPT-4 Turbo",
                context_length=128000,
                capabilities={"chat", "completion"},
                description="Latest GPT-4 model with increased capabilities and improved instruction following"
            ),
            "gpt-4": OpenAIModel(
                id="gpt-4",
                name="GPT-4",
                context_length=8192,
                capabilities={"chat", "completion"},
                description="Advanced model that can understand and generate text, code, and analyze images"
            ),
            
            # 嵌入模型
            "text-embedding-3-small": OpenAIModel(
                id="text-embedding-3-small",
                name="Text Embedding 3 Small",
                context_length=8191,
                capabilities={"embedding"},
                description="Smaller, cost-effective embedding model with good performance"
            ),
            "text-embedding-3-large": OpenAIModel(
                id="text-embedding-3-large",
                name="Text Embedding 3 Large",
                context_length=8191,
                capabilities={"embedding"},
                description="Latest embedding model with improved performance across tasks"
            ),
        }
        
        self._default_model = "gpt-3.5-turbo"
        self._default_embedding_model = "text-embedding-3-small"
        
        logger.info("Initialized OpenAI provider")
    
    @property
    def name(self) -> str:
        """獲取提供商名稱。"""
        return "openai"
    
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
            raise ValueError(f"Model '{model_id}' is not supported by OpenAI provider")
        
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
            logger.error(f"OpenAI chat error: {str(e)}")
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
            model: 要使用的模型ID，如果為None則使用預設嵌入模型
            **kwargs: 額外的參數
            
        Returns:
            Union[List[float], List[List[float]]]: 文字的嵌入向量
        """
        model_id = model or self._default_embedding_model
        model_obj = self.get_model(model_id)  # 驗證模型是否支援
        
        if "embedding" not in model_obj.capabilities:
            raise ValueError(f"Model '{model_id}' does not support embedding capability")
        
        # 確保文字以列表形式提供
        input_texts = [text] if isinstance(text, str) else text
        
        logger.debug(f"Embedding request to {model_id} with {len(input_texts)} texts")
        
        start_time = time.time()
        try:
            response = self._client.embeddings.create(
                model=model_id,
                input=input_texts,
                **kwargs
            )
            
            embeddings = [item.embedding for item in response.data]
            
            # 如果只嵌入一個文字，返回單個向量
            if isinstance(text, str):
                return embeddings[0]
            
            return embeddings
        
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
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