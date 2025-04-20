"""測試環境變數載入"""

import os
import sys
from pathlib import Path

def main():
    print(f"Python版本：{sys.version}")
    print(f"當前工作目錄：{os.getcwd()}")
    
    # 檢查是否安裝了dotenv
    try:
        from dotenv import load_dotenv
        print("dotenv已安裝，版本：", end="")
        import pkg_resources
        print(pkg_resources.get_distribution("python-dotenv").version)
        
        # 檢查.env文件是否存在
        env_path = Path(".env")
        if env_path.exists():
            print(f".env文件存在：{env_path.absolute()}")
            
            # 加載.env文件
            load_dotenv()
            print("已載入.env文件")
            
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
            print(f".env文件不存在，將創建示例文件：{env_path.absolute()}")
            with open(".env.example", "w", encoding="utf-8") as f:
                f.write("""# LLM API金鑰配置
# 複製此文件為.env並填入您的API金鑰

# 默認提供商設置
DEFAULT_LLM_PROVIDER=xai

# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORGANIZATION=your_organization_id_here

# DeepSeek API配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# X.AI API配置
XAI_API_KEY=your_xai_api_key_here

# Anthropic API配置
ANTHROPIC_API_KEY=your_anthropic_api_key_here
""")
            print("已創建.env.example示例文件")
    except ImportError:
        print("python-dotenv未安裝，請使用以下命令安裝：")
        print("pip install python-dotenv")

if __name__ == "__main__":
    main()
