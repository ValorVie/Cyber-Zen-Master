"""LLM客戶端封裝，提供與LLM互動的統一介面。"""

import os
from typing import Dict, List, Any, Optional

from llm.client import LLMClient
from llm.providers.deepseek import DeepSeekProvider
from llm.providers.openai import OpenAIProvider
from llm.providers.external_api import ExternalAPIProvider
from llm.config.loader import ConfigLoader
from llm.services.cache import LLMCache
from llm.services.metrics import LLMMetrics


def create_llm_client() -> LLMClient:
    """創建並配置LLM客戶端。
    
    從配置文件或環境變數加載API金鑰和其他設定，然後創建一個
    配置好的LLMClient實例。
    
    Returns:
        LLMClient: 配置好的LLM客戶端實例
    """
    # 加載配置
    config_loader = ConfigLoader()
    config = config_loader.load()
    
    # 創建服務
    cache = LLMCache()
    metrics = LLMMetrics()
    
    # 創建客戶端
    client = LLMClient(
        cache=cache,
        metrics=metrics,
        retry_attempts=3,
        enable_fallback=True
    )
    
    # 註冊DeepSeek提供商（如果配置了）
    deepseek_config = config.get("deepseek", {})
    if deepseek_config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY"):
        api_key = deepseek_config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        base_url = deepseek_config.get("base_url", "https://api.deepseek.com/v1")
        
        deepseek_provider = DeepSeekProvider(
            api_key=api_key,
            base_url=base_url
        )
        client.register_provider("deepseek", deepseek_provider)
    
    # 註冊OpenAI提供商（如果配置了）
    openai_config = config.get("openai", {})
    if openai_config.get("api_key") or os.environ.get("OPENAI_API_KEY"):
        api_key = openai_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        base_url = openai_config.get("base_url", "https://api.openai.com/v1")
        organization = openai_config.get("organization") or os.environ.get("OPENAI_ORGANIZATION")
        
        openai_provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            organization=organization
        )
        client.register_provider("openai", openai_provider)
    
    # 註冊外部API提供商（如果配置了）
    external_config = config.get("external_api", {})
    if external_config.get("endpoints"):
        external_provider = ExternalAPIProvider(
            endpoints=external_config.get("endpoints", {})
        )
        client.register_provider("external", external_provider)
    
    # 設置預設提供商
    default_provider = config_loader.get_default_provider()
    if default_provider and default_provider in client.list_providers():
        client.set_default_provider(default_provider)
    
    return client


class LLMClient:
    """LLM客戶端封裝，保持與現有代碼的兼容性。
    
    這個類為現有代碼提供了一個兼容層，讓它可以繼續使用舊的介面。
    """
    
    def __init__(self, api_key=None, base_url=None):
        """初始化LLM客戶端。
        
        Args:
            api_key: API金鑰（可選，如果指定則覆蓋配置）
            base_url: API基礎URL（可選，如果指定則覆蓋配置）
        """
        # 加載配置
        config_loader = ConfigLoader()
        config = config_loader.load()
        
        # 如果指定了API金鑰，更新配置
        if api_key:
            config_loader.update_provider_config("deepseek", {"api_key": api_key})
            if base_url:
                config_loader.update_provider_config("deepseek", {"base_url": base_url})
            config_loader.set_default_provider("deepseek")
            config_loader.save()
        
        # 創建新的客戶端
        self.client = create_llm_client()
    
    def chat(self, messages, temperature=0.5, stream=True):
        """與LLM交互，保持與現有代碼的兼容性。
        
        Args:
            messages: 消息列表
            temperature: 溫度參數
            stream: 是否以串流方式返回回應
            
        Returns:
            str: LLM回應內容
        """
        try:
            response = self.client.chat(
                messages=messages,
                temperature=temperature,
                stream=stream,
                verbose=stream  # 保持與原始代碼一致，在流式模式下打印輸出
            )
            
            return response.content
                
        except Exception as e:
            print(f"LLM調用出錯: {str(e)}")
            raise


# 使用示例
if __name__ == "__main__":
    llm = LLMClient()
    messages = [
        {"role": "user", "content": "你好"}
    ]
    response = llm.chat(messages)
    print(f"響應: {response}")