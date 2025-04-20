"""LLM 回應快取服務。"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

# 設置日誌
logger = logging.getLogger(__name__)


class LLMCache:
    """LLM 回應快取服務。
    
    用於快取LLM回應以減少API呼叫並提高效能。
    """
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        """初始化LLM快取。
        
        Args:
            ttl: 快取項的存活時間（秒），預設1小時
            max_size: 快取的最大項數，預設1000
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}  # 映射鍵到(值, 過期時間)的元組
        self._ttl = ttl
        self._max_size = max_size
    
    def _generate_key(self, data: Any) -> str:
        """從輸入數據生成快取鍵。
        
        Args:
            data: 用於生成鍵的輸入數據
            
        Returns:
            str: 生成的快取鍵
        """
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """從快取中獲取值。
        
        Args:
            key: 快取鍵或用於生成鍵的數據
            
        Returns:
            Any: 與鍵關聯的值，如果鍵不存在或已過期則為None
        """
        cache_key = key if isinstance(key, str) else self._generate_key(key)
        
        if cache_key not in self._cache:
            return None
        
        value, expiry = self._cache[cache_key]
        if expiry < time.time():
            # 快取項已過期
            del self._cache[cache_key]
            return None
        
        logger.debug(f"Cache hit for key: {cache_key[:8]}...")
        return value
    
    def set(self, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        """在快取中設置值。
        
        Args:
            key: 快取鍵或用於生成鍵的數據
            value: 要快取的值
            ttl: 此項的自定義TTL（秒），如果為None則使用預設TTL
        """
        cache_key = key if isinstance(key, str) else self._generate_key(key)
        expiry = time.time() + (ttl if ttl is not None else self._ttl)
        
        # 如果快取已滿，移除最舊的項
        if len(self._cache) >= self._max_size and cache_key not in self._cache:
            oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
            del self._cache[oldest_key]
        
        self._cache[cache_key] = (value, expiry)
        logger.debug(f"Cached value for key: {cache_key[:8]}...")
    
    def has(self, key: Any) -> bool:
        """檢查快取中是否存在有效的值。
        
        Args:
            key: 快取鍵或用於生成鍵的數據
            
        Returns:
            bool: 如果快取中存在有效值，則為True，否則為False
        """
        return self.get(key) is not None
    
    def invalidate(self, key: Any) -> None:
        """從快取中移除項。
        
        Args:
            key: 快取鍵或用於生成鍵的數據
        """
        cache_key = key if isinstance(key, str) else self._generate_key(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug(f"Invalidated cache for key: {cache_key[:8]}...")
    
    def clear(self) -> None:
        """清空整個快取。"""
        self._cache.clear()
        logger.debug("Cache cleared")
    
    def cleanup(self) -> int:
        """清理過期的快取項。
        
        Returns:
            int: 清理的項數
        """
        current_time = time.time()
        expired_keys = [k for k, (_, exp) in self._cache.items() if exp < current_time]
        
        for key in expired_keys:
            del self._cache[key]
        
        count = len(expired_keys)
        if count > 0:
            logger.debug(f"Cleaned up {count} expired cache entries")
        
        return count
    
    @property
    def size(self) -> int:
        """獲取當前快取大小。
        
        Returns:
            int: 快取中的項數
        """
        return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """獲取快取統計資訊。
        
        Returns:
            Dict[str, Any]: 包含快取統計資料的字典
        """
        current_time = time.time()
        expired = sum(1 for _, exp in self._cache.values() if exp < current_time)
        active = len(self._cache) - expired
        
        return {
            "total": len(self._cache),
            "active": active,
            "expired": expired,
            "max_size": self._max_size,
            "ttl": self._ttl,
            "utilization": (active / self._max_size) if self._max_size > 0 else 0
        }