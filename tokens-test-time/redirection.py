"""
Test-Time Register Redirection Hook
===================================
Implements the active intervention mechanism from Stage 2.3 (Jiang et al., NeurIPS 2025).
For identified candidate register neurons in a target transformer block's MLP,
the hook redirects outlier-token activations into an empty register slot (token index 1)
and resets the original outlier positions to the neuron's spatial mean.

Public Interface:
- `make_redirect_hook`: Factory function creating the PyTorch forward hook.
- `TestTimeRegisterManager`: Context manager and lifecycle controller for the intervention hook.
"""

import logging
from typing import Callable, List, Optional
import torch
import torch.nn as nn
from src.metrics.outlier_mask import get_outlier_mask

logger = logging.getLogger(__name__)


def make_redirect_hook(
    neuron_ids: List[int],
    outlier_mask_fn: Callable[[torch.Tensor, int], torch.Tensor] = get_outlier_mask,
    k_registers: int = 1,
) -> Callable:
    """
    Creates a PyTorch forward hook for block.mlp.act that:
    1. Identifies outlier patch tokens using outlier_mask_fn.
    2. For each identified neuron, accumulates outlier activation mass into
       the register token slot at index 1.
    3. Resets the original outlier patch positions to the per-sample spatial mean
       for that neuron, matching Jiang et al.'s 'mean' normal_values strategy.

    :param neuron_ids: List of integer neuron indices to redirect.
    :param outlier_mask_fn: Callable accepting (activations, k_registers) returning [B, N] bool.
    :param k_registers: Number of register slots in the sequence layout (default: 1).
    :return: Hook function (module, inp, out) -> modified_out.
    """
    def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> torch.Tensor:
        # out shape: [B, S, hidden_dim], where S = 1 (CLS) + k_registers + N (patches)
        modified = out.clone()
        start_idx = 1 + k_registers
        patch_acts = out[:, start_idx:, :]  # [B, N, hidden_dim]

        # Compute per-token boolean outlier mask over spatial patches
        mask = outlier_mask_fn(out, k_registers=k_registers)  # [B, N] boolean

        for neuron_id in neuron_ids:
            neuron_vals = patch_acts[:, :, neuron_id]              # [B, N]
            neuron_mean = neuron_vals.mean(dim=-1, keepdim=True)   # [B, 1]

            # Accumulate outlier activity into register slot at index 1
            outlier_vals = torch.where(mask, neuron_vals, torch.zeros_like(neuron_vals))
            # Sum outlier activation across patches and deposit into register slot (index 1)
            modified[:, 1, neuron_id] = outlier_vals.sum(dim=-1)

            # Reset original outlier positions to the spatial mean for that neuron
            reset_vals = torch.where(mask, neuron_mean.expand_as(neuron_vals), neuron_vals)
            modified[:, start_idx:, neuron_id] = reset_vals

        return modified

    return hook


class TestTimeRegisterManager:
    """
    Context manager and lifecycle controller for attaching and removing
    test-time redirection forward hooks on a target transformer block.
    """

    def __init__(
        self,
        model: nn.Module,
        target_block: int,
        neuron_ids: List[int],
        k_registers: int = 1,
    ):
        """
        :param model: Target RegisterVisionTransformer model.
        :param target_block: Block index to attach hook onto.
        :param neuron_ids: List of neuron indices to redirect.
        :param k_registers: Number of register slots in sequence.
        """
        self.model = model
        self.target_block = target_block
        self.neuron_ids = neuron_ids
        self.k_registers = k_registers
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _get_target_module(self) -> nn.Module:
        blocks = getattr(self.model, "blocks", None)
        if blocks is None and hasattr(self.model, "base_model"):
            blocks = self.model.base_model.blocks
        if blocks is None:
            raise AttributeError("Could not find .blocks in model.")

        if not (0 <= self.target_block < len(blocks)):
            raise IndexError(f"target_block {self.target_block} out of bounds.")

        block = blocks[self.target_block]
        if not hasattr(block, "mlp") or not hasattr(block.mlp, "act"):
            raise AttributeError(f"Block {self.target_block} has no .mlp.act submodule.")
        return block.mlp.act

    def attach(self) -> None:
        """Attaches the redirection hook onto the target block's MLP activation."""
        if self.handle is not None:
            self.remove()

        target_module = self._get_target_module()
        hook_fn = make_redirect_hook(
            neuron_ids=self.neuron_ids,
            outlier_mask_fn=get_outlier_mask,
            k_registers=self.k_registers,
        )
        self.handle = target_module.register_forward_hook(hook_fn)
        logger.info(
            "Attached test-time redirection hook onto block %d.mlp.act for neurons %s.",
            self.target_block,
            self.neuron_ids,
        )

    def remove(self) -> None:
        """Removes the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            logger.info("Removed test-time redirection hook from block %d.", self.target_block)

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False
