"""測試環境變數載入（簡化版本）"""

import os
import sys
from pathlib import Path
import re

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

def main():
    print(f"Python版本：{sys.version}")
    print(f"當前工作目錄：{os.getcwd()}")
    
    # 檢查.env文件是否存在
    env_path = Path(".env")
    if env_path.exists():
        print(f".env文件存在：{env_path.absolute()}")
        
        # 解析.env文件
        env_vars = parse_dotenv(env_path)
        print(f"已從.env加載 {len(env_vars)} 個環境變數")
        
        # 查看環境變數
        print("\n環境變數：")
        for key in ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY", "DEFAULT_LLM_PROVIDER"]:
            value = os.environ.get(key, "未設置")
            if value != "未設置" and len(value) > 8:
                # 遮蔽API金鑰
                display_value = value[:8] + "..." + value[-4:] if len(value) > 12 else value[:4] + "..."
                print(f"{key}: {display_value}")
            else:
                print(f"{key}: {value}")
    else:
        print(f".env文件不存在：{env_path.absolute()}")
        print("已創建.env.example示例文件")

if __name__ == "__main__":
    main()
