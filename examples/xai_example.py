"""X.AI Grok 模型使用示例。"""

import os
import sys
import logging

# 將父目錄添加到路徑，以便可以導入llm包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client import LLMClient

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函數。"""
    # 檢查是否設置了X.AI API金鑰
    if not os.environ.get("XAI_API_KEY"):
        print("請設置XAI_API_KEY環境變數以使用X.AI Grok模型")
        print("例如：export XAI_API_KEY=your_api_key")
        return
    
    print("初始化X.AI LLM客戶端...")
    llm = LLMClient(
        api_key=os.environ.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    
    print("\n=== 基本聊天示例 ===")
    
    # 示例1：基本問答
    messages = [
        {"role": "user", "content": "你好，請介紹一下你自己，你是誰？"}
    ]
    
    print("\n問題: 你好，請介紹一下你自己，你是誰？")
    print("回答中...")
    response = llm.chat(messages, temperature=0.7)
    print(f"回答: {response}")
    
    # 示例2：更復雜的問題
    messages = [
        {"role": "user", "content": "請解釋量子糾纏的概念，並舉一個簡單的例子。"}
    ]
    
    print("\n問題: 請解釋量子糾纏的概念，並舉一個簡單的例子。")
    print("回答中...")
    response = llm.chat(messages, temperature=0.7)
    print(f"回答: {response}")
    
    # 示例3：創意寫作
    messages = [
        {"role": "user", "content": "請以人工智能和人類共存的主題，寫一段未來世界的場景描述，大約100字。"}
    ]
    
    print("\n問題: 請以人工智能和人類共存的主題，寫一段未來世界的場景描述，大約100字。")
    print("回答中...")
    response = llm.chat(messages, temperature=0.9)
    print(f"回答: {response}")
    
    # 示例4：多輪對話
    messages = [
        {"role": "user", "content": "你能幫我寫一個簡單的Python函數嗎？我想要一個函數，可以計算斐波那契數列的第n項。"},
        {"role": "assistant", "content": "當然可以，以下是一個計算斐波那契數列的第n項的Python函數：\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n+1):\n            a, b = b, a + b\n        return b\n```\n\n這個函數使用迭代方法計算斐波那契數列，比遞歸方法更高效。你可以這樣使用它：\n\n```python\nprint(fibonacci(10))  # 輸出: 55\n```"},
        {"role": "user", "content": "這看起來很好，但我想要一個更高效的版本。你能使用矩陣方法實現嗎？"}
    ]
    
    print("\n問題: 這看起來很好，但我想要一個更高效的版本。你能使用矩陣方法實現嗎？")
    print("回答中...")
    response = llm.chat(messages, temperature=0.5)
    print(f"回答: {response}")
    
    # 示例5：串流模式
    messages = [
        {"role": "user", "content": "請解釋人工智能中的大型語言模型（LLM）是如何工作的。"}
    ]
    
    print("\n問題: 請解釋人工智能中的大型語言模型（LLM）是如何工作的。")
    print("回答（串流模式）:")
    
    # 在串流模式下回應會被直接打印出來
    response = llm.chat(messages, temperature=0.7, stream=True)
    
    print("\n\n完整回應長度:", len(response))


if __name__ == "__main__":
    main()