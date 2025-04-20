"""LLM架構使用示例。"""

import os
import sys
import logging
from typing import Dict, Any

# 將父目錄添加到路徑，以便可以導入llm包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.client import LLMClient
from llm.providers.deepseek import DeepSeekProvider
from llm.providers.openai import OpenAIProvider
from llm.providers.external_api import ExternalAPIProvider, ResponseTransformer
from llm.services.cache import LLMCache
from llm.services.metrics import LLMMetrics


# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_client() -> LLMClient:
    """設置LLM客戶端。
    
    Returns:
        LLMClient: 配置好的LLM客戶端
    """
    # 創建服務
    cache = LLMCache(ttl=3600, max_size=1000)
    metrics = LLMMetrics()
    
    # 創建客戶端
    client = LLMClient(
        cache=cache,
        metrics=metrics,
        retry_attempts=3,
        enable_fallback=True
    )
    
    # 註冊DeepSeek提供商
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        deepseek_provider = DeepSeekProvider(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )
        client.register_provider("deepseek", deepseek_provider)
        logger.info("Registered DeepSeek provider")
    
    # 註冊OpenAI提供商
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        openai_provider = OpenAIProvider(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1"
        )
        client.register_provider("openai", openai_provider)
        logger.info("Registered OpenAI provider")
    
    # 註冊外部API提供商（示例使用虛構的搜索API）
    # 在實際應用中，您需要替換為真實的API端點和認證資訊
    search_api_key = os.environ.get("SEARCH_API_KEY")
    if search_api_key:
        external_provider = ExternalAPIProvider({
            "search": {
                "url": "https://api.example.com/search",
                "method": "POST",
                "headers": {"Authorization": f"Bearer {search_api_key}"},
                "response_path": "results"
            }
        })
        client.register_provider("external", external_provider)
        logger.info("Registered External API provider")
    
    # 設置預設提供商（優先使用OpenAI，其次是DeepSeek）
    if "openai" in client.list_providers():
        client.set_default_provider("openai")
    elif "deepseek" in client.list_providers():
        client.set_default_provider("deepseek")
    
    return client


def example_chat(client: LLMClient) -> None:
    """示例：基本聊天。
    
    Args:
        client: LLM客戶端
    """
    print("\n=== 基本聊天示例 ===")
    messages = [
        {"role": "user", "content": "你好，請幫我寫一首關於AI的短詩。"}
    ]
    
    # 使用預設提供商
    response = client.chat(messages=messages, temperature=0.7)
    print(f"預設提供商回應:\n{response.content}\n")
    
    # 如果有多個提供商，嘗試使用不同提供商
    providers = client.list_providers()
    if len(providers) > 1 and "deepseek" in providers:
        response = client.chat(messages=messages, provider="deepseek", temperature=0.7)
        print(f"DeepSeek回應:\n{response.content}\n")


def example_streaming_chat(client: LLMClient) -> None:
    """示例：串流聊天。
    
    Args:
        client: LLM客戶端
    """
    print("\n=== 串流聊天示例 ===")
    messages = [
        {"role": "user", "content": "描述台灣的美食文化，至少列出5種著名的台灣小吃。"}
    ]
    
    print("串流回應：")
    response = client.chat(
        messages=messages, 
        temperature=0.7,
        stream=True,
        verbose=True  # 打印串流輸出
    )
    print(f"\n完整回應（長度：{len(response.content)}字符）")


def example_model_selection(client: LLMClient) -> None:
    """示例：選擇不同的模型。
    
    Args:
        client: LLM客戶端
    """
    print("\n=== 模型選擇示例 ===")
    messages = [
        {"role": "system", "content": "你是一個有用的AI助手。"},
        {"role": "user", "content": "請簡要解釋量子電腦的工作原理。"}
    ]
    
    if "openai" in client.list_providers():
        try:
            # 使用GPT-4
            response = client.chat(
                messages=messages,
                provider="openai",
                model="gpt-4",
                temperature=0.5
            )
            print(f"GPT-4回應:\n{response.content}\n")
            
            # 使用GPT-3.5
            response = client.chat(
                messages=messages,
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.5
            )
            print(f"GPT-3.5回應:\n{response.content}\n")
        
        except Exception as e:
            print(f"模型選擇示例出錯: {str(e)}")


def example_external_api(client: LLMClient) -> None:
    """示例：使用外部API。
    
    Args:
        client: LLM客戶端
    """
    print("\n=== 外部API示例 ===")
    
    if "external" in client.list_providers():
        try:
            # 呼叫搜索API（示例）
            search_results = client.call(
                provider="external",
                endpoint="search",
                params={"query": "quantum computing advances", "limit": 3}
            )
            print(f"搜索結果:\n{search_results}\n")
            
            # 結合LLM使用搜索結果（示例）
            if search_results and "openai" in client.list_providers():
                prompt = f"基於以下搜索結果，總結量子計算的最新進展：\n{search_results}"
                response = client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    provider="openai",
                    temperature=0.3
                )
                print(f"基於搜索結果的總結:\n{response.content}\n")
        
        except Exception as e:
            print(f"外部API示例出錯: {str(e)}")
            print("注意：這個示例需要實際的API端點才能正常工作。")


def example_metrics(client: LLMClient) -> None:
    """示例：查看使用指標。
    
    Args:
        client: LLM客戶端
    """
    print("\n=== 使用指標示例 ===")
    
    # 執行一些請求
    client.chat(
        messages=[{"role": "user", "content": "你好"}],
        temperature=0.5
    )
    
    client.chat(
        messages=[{"role": "user", "content": "今天天氣如何？"}],
        temperature=0.5
    )
    
    # 獲取指標
    stats = client.metrics.get_stats()
    print(f"總請求數: {stats['total_requests']}")
    print(f"平均延遲: {stats['avg_latency']:.4f}秒")
    if 'total_tokens' in stats:
        print(f"總Token數: {stats['total_tokens']}")
    
    # 獲取提供商指標
    for provider in client.list_providers():
        provider_stats = client.metrics.get_provider_stats(provider)
        print(f"\n{provider}提供商:")
        print(f"  請求數: {provider_stats.get('requests', 0)}")
        print(f"  平均延遲: {provider_stats.get('avg_latency', 0):.4f}秒")


def example_caching(client: LLMClient) -> None:
    """示例：使用快取。
    
    Args:
        client: LLM客戶端
    """
    print("\n=== 快取示例 ===")
    
    messages = [
        {"role": "user", "content": "請解釋什麼是人工智能？"}
    ]
    
    print("第一次請求（未快取）：")
    start_time = __import__('time').time()
    response1 = client.chat(messages=messages, use_cache=True)
    time1 = __import__('time').time() - start_time
    
    print(f"請求時間: {time1:.4f}秒")
    
    print("\n第二次請求（已快取）：")
    start_time = __import__('time').time()
    response2 = client.chat(messages=messages, use_cache=True)
    time2 = __import__('time').time() - start_time
    
    print(f"請求時間: {time2:.4f}秒")
    print(f"速度提升: {time1/time2 if time2 > 0 else 'N/A'}倍")
    print(f"回應相同: {response1.content == response2.content}")
    
    # 快取統計
    stats = client.cache.stats()
    print(f"\n快取統計:")
    print(f"  活躍項數: {stats['active']}")
    print(f"  最大容量: {stats['max_size']}")
    print(f"  使用率: {stats['utilization']:.2%}")


def main():
    """主函數。"""
    # 設置LLM客戶端
    client = setup_client()
    
    # 檢查是否有提供商可用
    if not client.list_providers():
        print("錯誤：未註冊任何LLM提供商。請設置必要的API金鑰。")
        print("您可以設置以下環境變數之一：")
        print("  - DEEPSEEK_API_KEY")
        print("  - OPENAI_API_KEY")
        return
    
    # 執行示例
    example_chat(client)
    
    while True:
        print("\n=== LLM架構示例 ===")
        print("1. 基本聊天")
        print("2. 串流聊天")
        print("3. 模型選擇")
        print("4. 外部API")
        print("5. 使用指標")
        print("6. 快取示例")
        print("0. 退出")
        
        choice = input("\n請選擇示例（0-6）: ")
        
        if choice == '0':
            break
        elif choice == '1':
            example_chat(client)
        elif choice == '2':
            example_streaming_chat(client)
        elif choice == '3':
            example_model_selection(client)
        elif choice == '4':
            example_external_api(client)
        elif choice == '5':
            example_metrics(client)
        elif choice == '6':
            example_caching(client)
        else:
            print("無效的選擇，請重試。")


if __name__ == "__main__":
    main()