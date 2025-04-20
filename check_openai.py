"""檢查OpenAI包的安裝狀態"""

import os
import sys

def main():
    print(f"Python版本：{sys.version}")
    print(f"當前工作目錄：{os.getcwd()}")
    
    try:
        import openai
        print(f"OpenAI包已安裝，版本：{openai.__version__}")
        print(f"OpenAI包路徑：{openai.__file__}")
    except ImportError:
        print("OpenAI包未安裝")
        print("嘗試使用pip安裝...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
            import openai
            print(f"安裝成功！OpenAI包版本：{openai.__version__}")
        except Exception as e:
            print(f"安裝失敗：{str(e)}")

if __name__ == "__main__":
    main()
