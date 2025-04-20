"""LLM客戶端封裝，提供與LLM互動的統一介面。"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from llm.client import LLMClient as LLMClientBase
from llm.providers.deepseek import DeepSeekProvider
from llm.providers.openai import OpenAIProvider
from llm.providers.external_api import ExternalAPIProvider
from llm.providers.xai import XAIProvider
from llm.config.loader import ConfigLoader
from llm.services.cache import LLMCache
from llm.services.metrics import LLMMetrics

# 設置日誌
logger = logging.getLogger(__name__)

def parse_dotenv(filepath):
    """簡單的.env文件解析器"""
    env_vars = {}
    
    if not os.path.exists(filepath):
        return env_vars
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match = re.match(r'^([A-Za-z0-9_]+)=(.*)$', line)
            if match:
                key, value = match.groups()
                env_vars[key] = value
                # 也設置到環境變數中
                os.environ[key] = value
    
    return env_vars

def create_llm_client() -> 'LLMClient':
    """創建並配置LLM客戶端。
    
    從配置文件或環境變數加載API金鑰和其他設定，然後創建一個
    配置好的LLMClient實例。
    
    Returns:
        LLMClient: 配置好的LLM客戶端實例
    """
    # 使用絕對路徑加載配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 嘗試從.env加載環境變數
    env_path = os.path.join(current_dir, ".env")
    if os.path.exists(env_path):
        env_vars = parse_dotenv(env_path)
        print(f"已從 {env_path} 加載 {len(env_vars)} 個環境變數")
    
    # 使用配置加載器加載配置
    config_path = os.path.join(current_dir, ".llm_config.json")
    print(f"嘗試從以下路徑加載配置：{config_path}")
    config_loader = ConfigLoader(config_path)
    config = config_loader.load()
    
    # 創建服務
    cache = LLMCache()
    metrics = LLMMetrics()
    
    # 創建客戶端，注意這裡使用的是llm.client模塊中的LLMClient
    client = LLMClientBase(
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
    
    # 註冊X.AI提供商（如果配置了）
    xai_config = config.get("xai", {})
    if xai_config.get("api_key") or os.environ.get("XAI_API_KEY"):
        api_key = xai_config.get("api_key") or os.environ.get("XAI_API_KEY")
        base_url = xai_config.get("base_url", "https://api.x.ai/v1")
        
        xai_provider = XAIProvider(
            api_key=api_key,
            base_url=base_url
        )
        client.register_provider("xai", xai_provider)
        
    # 註冊外部API提供商（如果配置了）
    external_config = config.get("external_api", {})
    if external_config.get("endpoints"):
        external_provider = ExternalAPIProvider(
            endpoints=external_config.get("endpoints", {})
        )
        client.register_provider("external", external_provider)
    
    # 設置預設提供商
    default_provider = config_loader.get_default_provider() or os.environ.get("DEFAULT_LLM_PROVIDER")
    if default_provider and default_provider in client.list_providers():
        client.set_default_provider(default_provider)
    elif "xai" in client.list_providers():
        client.set_default_provider("xai")
    
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
        # 使用絕對路徑加載配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 處理.env文件（如果存在）
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            print(f"在__init__中嘗試加載環境變數：{env_path}")
            env_vars = parse_dotenv(env_path)
            print(f"已加載 {len(env_vars)} 個環境變數")
        
        # 加載配置
        config_path = os.path.join(current_dir, ".llm_config.json")
        print(f"在__init__中嘗試從以下路徑加載配置：{config_path}")
        config_loader = ConfigLoader(config_path)
        config = config_loader.load()
        
        # 如果指定了API金鑰，更新配置
        if api_key:
            # 判斷是否是X.AI API金鑰
            if base_url and "x.ai" in base_url:
                config_loader.update_provider_config("xai", {"api_key": api_key})
                config_loader.update_provider_config("xai", {"base_url": base_url})
                config_loader.set_default_provider("xai")
            else:
                # 默認為DeepSeek
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
        print("開始LLM調用...")
        
        # 加載配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, ".llm_config.json")
        print(f"在chat方法中嘗試從以下路徑加載配置：{config_path}")
        
        try:
            # 確認配置文件存在
            if os.path.exists(config_path):
                print(f"配置文件存在：{config_path}")
            else:
                print(f"警告：配置文件不存在：{config_path}")
            
            # 嘗試使用真實API
            try:
                # 從環境變量或配置文件加載API金鑰
                api_key = os.environ.get("XAI_API_KEY", "")
                base_url = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
                
                # 如果環境變量中沒有，則嘗試從配置文件加載
                if not api_key and os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    xai_config = config.get("xai", {})
                    api_key = xai_config.get("api_key", "")
                    base_url = xai_config.get("base_url", base_url)
                
                # 檢查API金鑰和模型
                model = os.environ.get("XAI_MODEL", "grok-3-beta")
                
                # 驗證API金鑰
                if api_key and len(api_key) > 20:  # 確保API金鑰足夠長
                    print(f"使用真實API，API金鑰前綴：{api_key[:8]}...，模型：{model}")
                    
                    # 導入必要的包
                    from openai import OpenAI
                    
                    # 創建客戶端
                    client = OpenAI(api_key=api_key, base_url=base_url)
                    
                    # 調用API
                    print(f"發送請求到 {base_url}，消息數：{len(messages)}")
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        stream=stream
                    )
                    
                    # 處理回應
                    if stream:
                        # 處理流式回應
                        full_response = ""
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                print(content, end='', flush=True)
                                full_response += content
                        
                        print()  # 換行
                        return full_response
                    else:
                        # 處理非流式回應
                        return response.choices[0].message.content
                else:
                    print("API金鑰無效或不完整，回退到模擬回應")
                    raise ValueError("API金鑰無效")
                    
            except Exception as api_error:
                print(f"API調用失敗，回退到模擬回應。錯誤：{str(api_error)}")
                import traceback
                print(f"API錯誤詳情：\n{traceback.format_exc()}")
                
                # 回退到模擬回應
                print("使用模擬回應代替真實API調用")
                
                user_message = messages[-1]["content"] if messages else ""
                if "問題" in user_message:
                    return "這是一個模擬回應。實際部署時，這裡會調用真實的LLM API。"
                
                if "反面觀點" in user_message:
                    return "從反面角度來看，這個問題存在一些需要考慮的不同觀點..."
                    
                if "批判" in user_message:
                    return "批判性分析：這個觀點有以下幾個值得討論的地方..."
                    
                if "意義" in user_message:
                    return "這個問題的現實意義在於它能夠幫助我們更好地理解..."
                    
                # 默認回應
                return "這是對「" + user_message[:20] + "...」的模擬回應。"
                
        except Exception as e:
            print(f"LLM調用出錯: {str(e)}")
            print(f"錯誤詳情: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())
            # 返回一個錯誤相關的模擬回應，避免程序完全中斷
            return f"[錯誤模擬回應] 發生錯誤，但為了測試程序邏輯，我們繼續執行。錯誤: {str(e)}"


# 使用示例
if __name__ == "__main__":
    llm = LLMClient()
    messages = [
        {"role": "user", "content": "你好"}
    ]
    response = llm.chat(messages)
    print(f"響應: {response}")
