"""
Register Vision Transformer (rViT) Wrapper
===========================================
Implements the RegisterVisionTransformer module for Track 1 research:
"Do Register Tokens Regularize Vision Transformers Under Data Scarcity?"

Mathematical Architecture & Sequencing
---------------------------------------
In standard Vision Transformers (Dosovitskiy et al., 2020), background patch tokens
often act as spurious "attention sinks" with abnormally high L2 activation norms.
Darcet et al. (ICLR 2024) introduce K learnable register tokens R to absorb these
artifacts.

Sequence Formulation:
1. Spatial Patch Embeddings & Official Positional Encoding:
   X_patch = PatchEmbed(Image) + E_pos_patch    in R^{B x N x d}   (N = 196)
   x_cls   = x_cls_param + E_pos_cls            in R^{B x 1 x d}
2. Learnable Register Token Prepending:
   R = [r_1, r_2, ..., r_K]                     in R^{1 x K x d}
   X_0 = [ x_cls || R || X_patch ]              in R^{B x (1 + K + N) x d}
   *Note*: Registers are prepended AFTER spatial positional embeddings are applied.
   Registers DO NOT receive spatial positional embeddings, ensuring they remain
   non-spatial memory scratchpads.
3. Transformer Blocks Execution:
   X_L = Blocks_{1..12}(X_0)                    in R^{B x (1 + K + N) x d}
4. Register Discard & Classification Head:
   x_cls_final = LayerNorm(X_L[:, 0, :])        in R^{B x d}
   y_hat       = LinearHead(x_cls_final)        in R^{B x num_classes}

Public Interface:
- `RegisterVisionTransformer`: The core PyTorch module wrapping a timm ViT backbone.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class RegisterVisionTransformer(nn.Module):
    """
    Vision Transformer wrapper that injects K learnable register tokens
    between the [CLS] token and spatial image patch tokens.

    Attributes:
        num_registers (int): Number of register tokens K (e.g. 0, 1, 4, 8).
        embed_dim (int): Hidden dimension size d (e.g. 192 for ViT-Tiny).
        base_model (nn.Module): The underlying timm VisionTransformer backbone.
        registers (nn.Parameter or None): Learnable tensor of shape [1, K, d].
        blocks (nn.Sequential): Forward reference to transformer blocks.
        head (nn.Module): Classification head.
    """

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        num_classes: int = 100,
        num_registers: int = 0,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        img_size: int = 224,
    ):
        """
        Initializes the RegisterVisionTransformer.

        :param model_name: Name of the timm Vision Transformer model architecture.
        :param num_classes: Number of classification targets (default 100 for CIFAR-100).
        :param num_registers: Number of register tokens K to inject (0, 1, 4, 8).
        :param pretrained: If True, loads ImageNet pretrained weights into the base backbone.
        :param drop_rate: Dropout rate for classifier head and patch projection.
        :param attn_drop_rate: Attention dropout rate in Multi-Head Self-Attention.
        :param img_size: Input spatial resolution in pixels (default 224).
        :raises ValueError: If num_registers is negative.
        """
        super().__init__()
        if num_registers < 0:
            raise ValueError(f"num_registers must be non-negative, got {num_registers}.")

        self.model_name = model_name
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.pretrained = pretrained
        self.img_size = img_size

        # 1. Instantiate base timm backbone
        # We load pretrained weights cleanly on the standard backbone WITHOUT registers
        # to avoid positional embedding dimension mismatch crashes in timm.
        self.base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            img_size=img_size,
        )

        self.embed_dim = self.base_model.embed_dim
        # Transparent block exposing for ViTAttentionHookManager and downstream tools
        self.blocks = self.base_model.blocks
        self.patch_embed = self.base_model.patch_embed
        self.head = self.base_model.head
        self.num_prefix_tokens = 1 + self.num_registers

        # 2. Add learnable register parameter
        if self.num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(1, num_registers, self.embed_dim))
            # Initialize with truncated normal distribution (std=0.02)
            # Crucial: Avoid zero-initialization to prevent dead attention gradient flatlines
            nn.init.trunc_normal_(self.registers, std=0.02)
            logger.info(
                "Initialized %d register tokens of dimension %d with trunc_normal_(std=0.02).",
                num_registers,
                self.embed_dim,
            )
        else:
            self.register_parameter("registers", None)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes feature projection, register token insertion, and transformer block encoding.

        :param x: Input image batch tensor of shape [B, 3, H, W].
        :return: Encoded hidden states of shape [B, 1 + K + N, d].
        """
        # Step 1: Compute patch embeddings and apply official learned positional embeddings
        x = self.base_model.patch_embed(x)
        x = self.base_model._pos_embed(x)  # Shape: [B, 1 (CLS) + N (Patches), d]

        # Step 2: Inject register tokens between [CLS] and spatial patches
        # Critical: Registers DO NOT receive spatial positional embeddings.
        if self.num_registers > 0 and self.registers is not None:
            cls_token = x[:, :1, :]      # [B, 1, d]
            patch_tokens = x[:, 1:, :]   # [B, N, d]
            reg_tokens = self.registers.expand(x.shape[0], -1, -1)  # [B, K, d]
            # Form full sequence: [CLS || Registers || Spatial Patches]
            x = torch.cat([cls_token, reg_tokens, patch_tokens], dim=1)  # [B, 1 + K + N, d]

        # Step 3: Optional patch drop & pre-transformer normalization
        if hasattr(self.base_model, "patch_drop") and self.base_model.patch_drop is not None:
            x = self.base_model.patch_drop(x)
        x = self.base_model.norm_pre(x)

        # Step 4: Pass through all 12 transformer blocks
        x = self.base_model.blocks(x)

        # Step 5: Final layer normalization
        x = self.base_model.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Discards register tokens and spatial patches, pooling solely the [CLS] token
        into the classification head.

        :param x: Hidden states from forward_features of shape [B, 1 + K + N, d].
        :param pre_logits: If True, returns features before the linear classifier.
        :return: Logits tensor [B, num_classes] or pre-logits features [B, d].
        """
        # Discard registers (indices 1..K) and spatial patches (indices 1+K..end)
        # Class token is strictly at index 0
        cls_out = x[:, 0]  # [B, d]

        if hasattr(self.base_model, "fc_norm") and self.base_model.fc_norm is not None:
            cls_out = self.base_model.fc_norm(cls_out)
        if hasattr(self.base_model, "head_drop") and self.base_model.head_drop is not None:
            cls_out = self.base_model.head_drop(cls_out)

        if pre_logits:
            return cls_out

        return self.base_model.head(cls_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass mapping input images directly to classification logits.

        :param x: Image batch tensor [B, 3, H, W].
        :return: Prediction logits [B, num_classes].
        """
        feats = self.forward_features(x)
        return self.forward_head(feats)

    # --------------------------------------------------------------------------
    # Sequence Token Partition Utilities (for Analysis & Visualization)
    # --------------------------------------------------------------------------

    def get_cls_token(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extracts the [CLS] token vector from encoded features.

        :param features: Output from forward_features [B, S, d].
        :return: [CLS] token tensor [B, d].
        """
        return features[:, 0, :]

    def get_register_tokens(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extracts register tokens from encoded features.

        :param features: Output from forward_features [B, S, d].
        :return: Register tokens tensor [B, K, d], or None if K=0.
        """
        if self.num_registers == 0:
            return None
        return features[:, 1 : (1 + self.num_registers), :]

    def get_spatial_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extracts spatial image patch tokens from encoded features.

        :param features: Output from forward_features [B, S, d].
        :return: Spatial patch tokens tensor [B, N, d] (N=196).
        """
        return features[:, (1 + self.num_registers) :, :]
