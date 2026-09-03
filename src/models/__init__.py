"""
Vision Transformer Models & Hook Instrumentation
=================================================
Provides architecture wrappers and diagnostic forward hook utilities for
Vision Transformers under the Track 1 research scope.

Public Exports:
- `ViTAttentionHookManager`: Non-invasive attention matrix and activation extractor.
"""

from .attention_hook import ViTAttentionHookManager

__all__ = [
    "ViTAttentionHookManager",
]
