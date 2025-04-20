"""LLM 模型基本介面。"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set


class LLMModel(ABC):
    """LLM 模型介面。
    
    定義所有LLM模型必須實作的屬性和方法。這包括模型識別碼、上下文長度和
    支援的功能等基本資訊。
    """
    
    @property
    @abstractmethod
    def id(self) -> str:
        """模型識別碼。"""
        pass
    
    @property
    @abstractmethod
    def context_length(self) -> int:
        """模型支援的最大上下文長度（以token為單位）。"""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> Set[str]:
        """模型支援的功能集合。
        
        可能的功能包括：
        - 'chat': 聊天功能
        - 'embedding': 嵌入功能
        - 'completion': 文本補全
        - 'function_calling': 函數呼叫
        - 'image_generation': 圖像生成
        - 'image_understanding': 圖像理解
        """
        pass
    
    def is_supported(self, feature: str) -> bool:
        """檢查模型是否支援指定功能。
        
        Args:
            feature: 要檢查的功能名稱
            
        Returns:
            bool: 如果模型支援該功能，則為True，否則為False
        """
        return feature in self.capabilities
    
    @property
    def name(self) -> str:
        """模型顯示名稱，可被子類覆寫以提供更友好的名稱。"""
        return self.id
    
    @property
    def description(self) -> Optional[str]:
        """模型描述，預設為None，可被子類覆寫。"""
        return None
    
    def __str__(self) -> str:
        """返回模型的字串表示。"""
        return f"{self.name} (context: {self.context_length} tokens)"
    
    def __repr__(self) -> str:
        """返回模型的程式表示。"""
        return f"{self.__class__.__name__}(id='{self.id}', context_length={self.context_length})"