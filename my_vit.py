from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import ensure_tuple_rep
from monai.networks.layers import Conv


class MyViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            pos_embed: str = "conv",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            post_activation="Sigmoid",
            qkv_bias: bool = False,
            mode_classification='cls',
        ) -> None:
            super().__init__()

            if not (0 <= dropout_rate <= 1):
                raise ValueError("dropout_rate should be between 0 and 1.")

            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size should be divisible by num_heads.")

            self.mode_classification = mode_classification
            self.classification = classification
            self.patch_embedding = PatchEmbeddingBlock(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                pos_embed=pos_embed,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
            )
            self.blocks = nn.ModuleList(
                [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for i in range(num_layers)]
            )
            self.norm = nn.LayerNorm(hidden_size)
            if self.classification:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
                if post_activation == "Tanh":
                    self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
                elif post_activation == "Sigmoid":
                    self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Sigmoid())
                else:
                    self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def get_activations_gradient(self):
        return self.gradients               

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations(self, x, mode='cls'):
        x = self.patch_embedding(x)
        return x

    def forward(self, x):
        x = self.patch_embedding(x)

        if self.mode_classification == 'cls':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        if self.mode_classification == 'cls':
            x = self.classification_head(x[:, 0])
        elif self.mode_classification == 'mean':
            x = self.classification_head(torch.mean(x, 1))

        return x, hidden_states_out


class MyVitAutoEnc(nn.Module):
    def __init__(self,
                in_channels: int,
                img_size: Union[Sequence[int], int],
                patch_size: Union[Sequence[int], int],
                out_channels: int = 1,
                deconv_chns: int = 16,
                hidden_size: int = 768,
                mlp_dim: int = 3072,
                num_layers: int = 12,
                num_heads: int = 12,
                pos_embed: str = "conv",
                qkv_bias: bool = False,
                dropout_rate: float = 0.0,
                spatial_dims: int = 3,
                num_classes: int = 1,
                classification=True,
                post_activation="Sigmoid",
                mode_classification='cls',
                ):
        super().__init__()
        self.gradients = []

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mode_classification = mode_classification
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            elif post_activation == "Sigmoid":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Sigmoid())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.spatial_dims = spatial_dims

        new_patch_size = [4] * self.spatial_dims
        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        self.conv3d_transpose = conv_trans(hidden_size,
                                           deconv_chns,
                                           kernel_size=new_patch_size,
                                           stride=new_patch_size)

        self.conv3d_transpose_1 = conv_trans(
            in_channels=deconv_chns,
            out_channels=out_channels,
            kernel_size=new_patch_size,
            stride=new_patch_size
        )

    def forward_autoenc(self, x):
        spatial_size = x.shape[2:]
        x = self.patch_embedding(x)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        hidden_norm = self.norm(x)

        x = torch.transpose(hidden_norm, 1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)

        return x, hidden_norm

    def forward(self, x):
        x = self.patch_embedding(x)

        if self.mode_classification == 'cls':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        if self.mode_classification == 'cls':
            x = self.classification_head(x[:, 0])
        elif self.mode_classification == 'mean':
            x = self.classification_head(torch.mean(x, 1))

        return x, hidden_states_out
