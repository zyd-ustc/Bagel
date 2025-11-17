from typing import Optional

import torch
from torch import nn
from .qwen2_navit import NaiveCache

class KeyValueAdapter(nn.Module):
    """
    Adapter for mapping key-value cache from understanding space to generation space.
    """
    def __init__(
        self, 
        num_layers: int, 
        num_heads: int, 
        head_dim: int,
        num_adapter_layers: Optional[int] = None,
    ):
        """
        Args:
            num_layers: Number of transformer layers (e.g., 32)
            num_heads: Number of key-value heads
            head_dim: Dimension of each attention head
            num_adapter_layers: Number of actual adapter layers to create.
                If None, creates one adapter per layer (no sharing).
                If < num_layers, multiple layers will share the same adapter.
                Example: num_layers=32, num_adapter_layers=8 means every 4 layers share one adapter.
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        if num_adapter_layers is None:
            num_adapter_layers = num_layers
        
        self.num_adapter_layers = num_adapter_layers
        
        self.layer_to_adapter_idx = [
            (i * num_adapter_layers) // num_layers 
            for i in range(num_layers)
        ]
        
        self.key_adapters = nn.ModuleList([
            nn.Linear(head_dim, head_dim, bias=False) 
            for _ in range(num_adapter_layers)
        ])
        self.value_adapters = nn.ModuleList([
            nn.Linear(head_dim, head_dim, bias=False)
            for _ in range(num_adapter_layers)
        ])
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize adapter weights to identity mapping"""
        for adapter_idx in range(self.num_adapter_layers):
            nn.init.eye_(self.key_adapters[adapter_idx].weight)
            nn.init.eye_(self.value_adapters[adapter_idx].weight)
    
    def forward(self, past_key_values: NaiveCache) -> NaiveCache:
        """
        Transform key-value cache from understanding space to generation space.
        
        Args:
            past_key_values: NaiveCache object containing keys and values for each layer
                Each key/value has shape: (seqlens, num_heads, head_dim)
            
        Returns:
            Transformed NaiveCache object with same structure
        """
        adapted_cache = NaiveCache(self.num_layers)
        
        for layer_idx in range(self.num_layers):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            
            if key_cache is None or value_cache is None:
                continue
            
            # key_cache shape: (seqlens, num_heads, head_dim)
            # value_cache shape: (seqlens, num_heads, head_dim)
            adapter_idx = self.layer_to_adapter_idx[layer_idx]
            
            adapted_key = self.key_adapters[adapter_idx](key_cache)
            adapted_value = self.value_adapters[adapter_idx](value_cache)
            
            adapted_cache.key_cache[layer_idx] = adapted_key
            adapted_cache.value_cache[layer_idx] = adapted_value
        
        return adapted_cache
