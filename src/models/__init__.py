"""
Vision Transformer Models & Hook Instrumentation
=================================================
Provides architecture wrappers and diagnostic forward hook utilities for
Vision Transformers under the Track 1 research scope:
"Do Register Tokens Regularize Vision Transformers Under Data Scarcity?"

Public Exports:
- `RegisterVisionTransformer`: Vision Transformer with K learnable register tokens.
- `ViTAttentionHookManager`: Non-invasive attention matrix and activation extractor.
"""

from .register_vit import RegisterVisionTransformer
from .attention_hook import ViTAttentionHookManager

__all__ = [
    "RegisterVisionTransformer",
    "ViTAttentionHookManager",
]
