"""
ViT Attention Hook Manager
===========================
Non-invasive forward hooks that intercept per-layer, per-head attention
weight matrices A^(l,h) from a timm Vision Transformer, WITHOUT breaking
backpropagation and WITHOUT modifying the model's forward pass.

Why we can't just read `output` in a plain forward hook:
    Modern timm builds (torch >= 2.0) use `F.scaled_dot_product_attention`
    (fused_attn=True) internally. That fused kernel never materializes the
    [B, H, S, S] softmax matrix -- it goes straight from Q,K,V to the output
    vectors. So the hook's `output` argument only contains the *result* of
    attention, not the attention weights themselves.

    To get A^(l,h), we re-run the exact same math the module would have run
    (qkv projection -> split heads -> scale -> softmax) using the module's
    OWN weights and the block's OWN input. This is mathematically identical
    to what happens inside `attn.forward`, so it is not an approximation --
    it is a faithful reconstruction, done once more, purely for measurement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ViTAttentionHookManager:
    """
    Attaches forward hooks to all multi-head attention layers in a timm
    Vision Transformer. Stores DETACHED, CPU-resident attention matrices
    and clears them on demand to prevent CUDA OOM.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self.intermediate_activations: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        # In timm ViT: model.blocks[i].attn
        for layer_idx, block in enumerate(self.model.blocks):
            hook = block.attn.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            self.hooks.append(hook)

    def _make_attn_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # `input` is a tuple; input[0] is the tensor that entered
            # attn.forward(x), i.e. the block's normed hidden state.
            x = input[0]

            with torch.no_grad():
                B, N, C = x.shape
                num_heads = module.num_heads
                head_dim = module.head_dim
                scale = module.scale

                # Reproduce module.qkv projection + head split exactly
                # as timm's Attention.forward does.
                qkv = module.qkv(x).reshape(
                    B, N, 3, num_heads, head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)  # each: [B, num_heads, N, head_dim]

                # timm applies q_norm / k_norm (Identity by default) before
                # scaling -- include them so results match exactly even if
                # a config enables QK-norm.
                q, k = module.q_norm(q), module.k_norm(k)

                q = q * scale
                attn = q @ k.transpose(-2, -1)          # [B, H, N, N]
                attn = attn.softmax(dim=-1)

                # Detach immediately and move off GPU: this is the single
                # most important line for avoiding CUDA OOM. Without it,
                # every stored tensor keeps the whole autograd graph for
                # that layer alive for the lifetime of self.attention_maps.
                self.attention_maps[layer_idx] = attn.detach().cpu()
                self.intermediate_activations[layer_idx] = x.detach().cpu()

        return hook_fn

    def clear(self):
        """Clears stored matrices from memory (call between batches/epochs)."""
        self.attention_maps.clear()
        self.intermediate_activations.clear()

    def remove(self):
        """Removes all PyTorch forward hooks. Always call this when done,
        e.g. in a `finally:` block, or hooks accumulate across repeated
        ViTAttentionHookManager(model) instantiations and silently multiply
        VRAM/CPU-RAM usage."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    # Convenience context-manager usage: `with ViTAttentionHookManager(m) as h:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        self.clear()
        return False
