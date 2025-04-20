"""配置加載器，用於加載LLM提供商的配置。"""

import json
import os
import logging
from typing import Dict, Any, Optional

# 設置日誌
logger = logging.getLogger(__name__)


class ConfigLoader:
    """LLM配置加載器。
    
    從文件或環境變數加載LLM提供商的配置。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置加載器。
        
        Args:
            config_path: 配置文件路徑，如果為None則使用預設路徑
        """
        self.config_path = config_path or os.path.join(os.path.expanduser("~"), ".llm_config.json")
        self._config: Dict[str, Any] = {}
        self._loaded = False
    
    def load(self, force_reload: bool = False) -> Dict[str, Any]:
        """加載配置。
        
        Args:
            force_reload: 是否強制重新加載
            
        Returns:
            Dict[str, Any]: 加載的配置
        """
        if self._loaded and not force_reload:
            return self._config
        
        # 從文件加載配置
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {self.config_path}: {str(e)}")
                self._config = {}
        
        # 從環境變數加載或覆蓋配置
        self._load_from_env()
        
        self._loaded = True
        return self._config
    
    def _load_from_env(self) -> None:
        """從環境變數加載配置。"""
        # OpenAI
        if "OPENAI_API_KEY" in os.environ:
            self._ensure_provider_config("openai")
            self._config["openai"]["api_key"] = os.environ["OPENAI_API_KEY"]
        
        if "OPENAI_ORGANIZATION" in os.environ:
            self._ensure_provider_config("openai")
            self._config["openai"]["organization"] = os.environ["OPENAI_ORGANIZATION"]
        
        # DeepSeek
        if "DEEPSEEK_API_KEY" in os.environ:
            self._ensure_provider_config("deepseek")
            self._config["deepseek"]["api_key"] = os.environ["DEEPSEEK_API_KEY"]
        
        # Anthropic
        if "ANTHROPIC_API_KEY" in os.environ:
            self._ensure_provider_config("anthropic")
            self._config["anthropic"]["api_key"] = os.environ["ANTHROPIC_API_KEY"]
        
        # 預設提供商
        if "DEFAULT_LLM_PROVIDER" in os.environ:
            self._config["default_provider"] = os.environ["DEFAULT_LLM_PROVIDER"]
    
    def _ensure_provider_config(self, provider: str) -> None:
        """確保提供商配置存在。
        
        Args:
            provider: 提供商名稱
        """
        if provider not in self._config:
            self._config[provider] = {}
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """獲取指定提供商的配置。
        
        Args:
            provider: 提供商名稱
            
        Returns:
            Dict[str, Any]: 提供商配置
        """
        if not self._loaded:
            self.load()
        
        return self._config.get(provider, {})
    
    def get_default_provider(self) -> Optional[str]:
        """獲取預設提供商。
        
        Returns:
            Optional[str]: 預設提供商名稱
        """
        if not self._loaded:
            self.load()
        
        return self._config.get("default_provider")
    
    def save(self, config: Optional[Dict[str, Any]] = None) -> None:
        """保存配置到文件。
        
        Args:
            config: 要保存的配置，如果為None則保存當前配置
        """
        to_save = config or self._config
        
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(to_save, f, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save configuration to {self.config_path}: {str(e)}")
            raise
    
    def update_provider_config(self, provider: str, config: Dict[str, Any]) -> None:
        """更新提供商配置。
        
        Args:
            provider: 提供商名稱
            config: 新的配置
        """
        if not self._loaded:
            self.load()
        
        if provider not in self._config:
            self._config[provider] = {}
        
        self._config[provider].update(config)
    
    def set_default_provider(self, provider: str) -> None:
        """設置預設提供商。
        
        Args:
            provider: 提供商名稱
        """
        if not self._loaded:
            self.load()
        
        self._config["default_provider"] = provider