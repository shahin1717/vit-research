"""
ViT Attention Hook Manager
===========================
Non-invasive PyTorch forward hooks for Vision Transformers (ViTs) that:
1. Faithfully intercept per-layer, per-head attention weight matrices A^(l,h)
   without modifying model execution or breaking backpropagation graphs.
2. Intercept intermediate residual-stream hidden states X^(l) across all blocks
   to enable accurate layerwise activation norm and outlier rate diagnostics.

Architecture & System Context
-----------------------------
Part of `src.models`, this module provides the measurement instrumentation for
Track 1 research ("Do Register Tokens Regularize Vision Transformers Under Data
Scarcity?"). Modern timm models (PyTorch >= 2.0) execute fused scaled dot-product
attention (`F.scaled_dot_product_attention`), bypassing physical allocation of
the [B, H, S, S] attention matrix. This manager reconstructs the exact softmax
attention weights from unnormalized Q/K projections while capturing unnormalized
block residual outputs (preventing LayerNorm norm squashing).

Public Interface
----------------
- `ViTAttentionHookManager`: Context manager and lifecycle handler for forward hooks.

Dependencies & Side Effects
---------------------------
- PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
- Registers PyTorch forward hooks onto `model.blocks[i].attn` and `model.blocks[i]`.
- All captured tensors are immediately `.detach().cpu()`-ed to eliminate VRAM retention.

Maintenance Invariants
----------------------
- Always invoke `remove()` or use context manager `with ...:` to unhook modules.
- Intermediate activations must be recorded at block output (the residual stream),
  NEVER inside `block.attn` where `LayerNorm` forces all token norms to sqrt(d).
"""

import logging
from typing import Dict, List, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ViTAttentionHookManager:
    """
    Attaches non-invasive forward hooks across all transformer blocks of a
    Vision Transformer model. Captures detached, CPU-resident attention matrices
    and block output residual hidden states while preventing CUDA OOM leaks.

    Attributes:
        model (nn.Module): The target Vision Transformer (or wrapped ViT).
        hooks (List[torch.utils.hooks.RemovableHandle]): Active PyTorch hook handles.
        attention_maps (Dict[int, torch.Tensor]): layer_idx -> [B, H, S, S] attention weights.
        intermediate_activations (Dict[int, torch.Tensor]): layer_idx -> [B, S, d] block outputs.
    """

    def __init__(self, model: nn.Module):
        """
        Initializes hook manager and attaches forward hooks to all blocks.

        :param model: A timm Vision Transformer model or custom wrapper containing blocks.
        :raises AttributeError: If no transformer blocks can be discovered.
        """
        self.model = model
        self.enabled: bool = True
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self.intermediate_activations: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _get_blocks(self) -> nn.ModuleList:
        """
        Discovers the transformer blocks sequence in standard timm models or
        nested model wrappers (e.g. RegisterVisionTransformer).

        :return: Iterable collection of transformer blocks.
        :raises AttributeError: If no blocks container can be found.
        """
        # 1. Direct attribute on model
        if hasattr(self.model, "blocks") and hasattr(self.model.blocks, "__iter__"):
            return self.model.blocks

        # 2. Check common wrapper attributes
        for candidate_attr in ["model", "vit", "backbone", "transformer"]:
            sub = getattr(self.model, candidate_attr, None)
            if sub is not None and hasattr(sub, "blocks") and hasattr(sub.blocks, "__iter__"):
                return sub.blocks

        raise AttributeError(
            f"Could not locate '.blocks' container in {type(self.model).__name__} "
            f"or its submodules ('model', 'vit', 'backbone', 'transformer')."
        )

    def _register_hooks(self) -> None:
        """
        Registers forward hooks on each block's attention module (for attention maps)
        and on the transformer block itself (for unnormalized residual activations).
        """
        blocks = self._get_blocks()
        for layer_idx, block in enumerate(blocks):
            # Hook attention module to faithfully reconstruct softmax attention weights
            attn_mod = getattr(block, "attn", None)
            if attn_mod is not None and hasattr(attn_mod, "qkv"):
                h_attn = attn_mod.register_forward_hook(self._make_attn_hook(layer_idx))
                self.hooks.append(h_attn)
            else:
                logger.warning("Block %d lacks expected .attn module with .qkv projection.", layer_idx)

            # Hook transformer block to capture output residual activations
            # Note: Must capture block output, NOT attn input, to prevent LayerNorm squashing
            h_block = block.register_forward_hook(self._make_block_hook(layer_idx))
            self.hooks.append(h_block)

    def _make_attn_hook(self, layer_idx: int):
        """
        Creates a hook function for an attention module to reconstruct A^(l,h).
        """
        def hook_fn(module, input_args, output):
            if not self.enabled:
                return
            x = input_args[0]
            with torch.no_grad():
                B, N, _ = x.shape
                num_heads = module.num_heads
                head_dim = module.head_dim
                scale = module.scale

                # Re-run identical projection math to unfused attention
                qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)

                # Apply Q/K normalization if present in architecture
                if hasattr(module, "q_norm") and callable(module.q_norm):
                    q = module.q_norm(q)
                if hasattr(module, "k_norm") and callable(module.k_norm):
                    k = module.k_norm(k)

                q = q * scale
                attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)

                # Detach and offload to CPU immediately to prevent VRAM accumulation
                self.attention_maps[layer_idx] = attn.detach().cpu()

        return hook_fn

    def _make_block_hook(self, layer_idx: int):
        """
        Creates a hook function for the block output to capture residual activations.
        """
        def hook_fn(module, input_args, output):
            if not self.enabled:
                return
            with torch.no_grad():
                # Block output tensor is the unnormalized residual stream [B, S, d]
                if isinstance(output, torch.Tensor):
                    self.intermediate_activations[layer_idx] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    self.intermediate_activations[layer_idx] = output[0].detach().cpu()

        return hook_fn

    def disable(self) -> None:
        """Disables hook execution without unregistering hook handles."""
        self.enabled = False

    def enable(self) -> None:
        """Enables hook execution."""
        self.enabled = True

    def clear(self) -> None:
        """
        Clears all stored attention matrices and activations from host RAM.
        Recommended to invoke between evaluation batches or epochs.
        """
        self.attention_maps.clear()
        self.intermediate_activations.clear()

    def remove(self) -> None:
        """
        Removes all registered PyTorch forward hooks.
        Prevents multiple duplicate hooks from accumulating on model blocks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatic hook removal and memory cleanup upon context exit."""
        self.remove()
        self.clear()
        return False

    def __del__(self):
        """Destructor cleanup guard against hook accumulation leaks."""
        self.remove()
