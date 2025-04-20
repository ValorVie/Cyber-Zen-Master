"""測試配置文件加載"""

import os
import json
from llm.config.loader import ConfigLoader

def main():
    # 使用絕對路徑加載配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, ".llm_config.json")
    
    print(f"嘗試從以下路徑加載配置：{config_path}")
    
    # 直接讀取文件
    if os.path.exists(config_path):
        print("配置文件存在！")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            print(f"配置文件原始內容：\n{config_content}")
            
            # 解析JSON
            try:
                config = json.loads(config_content)
                print("\n解析後的配置：")
                for key, value in config.items():
                    if key != "xai" or not isinstance(value, dict) or "api_key" not in value:
                        print(f"{key}: {value}")
                    else:
                        # 不顯示完整API金鑰
                        api_key = value.get("api_key", "")
                        api_key_masked = api_key[:8] + "..." if len(api_key) > 8 else api_key
                        print(f"{key}: {{api_key: {api_key_masked}, base_url: {value.get('base_url')}}}")
            except json.JSONDecodeError as e:
                print(f"JSON解析錯誤：{str(e)}")
    else:
        print(f"配置文件不存在：{config_path}")
    
    # 使用ConfigLoader
    print("\n使用ConfigLoader加載配置")
    config_loader = ConfigLoader(config_path)
    config = config_loader.load()
    
    print("ConfigLoader加載的配置：")
    for key, value in config.items():
        if key != "xai" or not isinstance(value, dict) or "api_key" not in value:
            print(f"{key}: {value}")
        else:
            # 不顯示完整API金鑰
            api_key = value.get("api_key", "")
            api_key_masked = api_key[:8] + "..." if len(api_key) > 8 else api_key
            print(f"{key}: {{api_key: {api_key_masked}, base_url: {value.get('base_url')}}}")

if __name__ == "__main__":
    main()
