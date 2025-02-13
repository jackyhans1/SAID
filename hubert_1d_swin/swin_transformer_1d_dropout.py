import math
import torch
import torch.nn as nn
import numpy as np
import json
import torch.nn.functional as F
# from torch.nn.utils.parametrizations import weight_norm
from torch.autograd import Variable

__all__ = [
    "Swin1D",
]

"""
 이 코드는 "https://github.com/yukara-ikemiya/Swin-Transformer-1d/blob/main/models/swin_transformer_1d_v2.py" 를
 기반으로 하였으며, 음성의 실제 길이(mask)를 고려하여 패딩된 부분은 어텐션 계산에서 마스킹하도록 일부 수정되었습니다.
"""

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows

def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): Length of data

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x

class PatchEmbed(nn.Module):
    def __init__(self, total_length=1280, win_size=16, num_channels=32, emb_size=128, norm_layer=True):
        super().__init__()
        self.total_length = total_length
        self.win_size = win_size
        self.num_channels = num_channels
        self.emb_size = emb_size
        self.norm_layer = norm_layer

        self.proj = nn.Linear(self.num_channels, self.emb_size)
        
        if self.norm_layer:
            self.norm = nn.LayerNorm(self.emb_size)

    def forward(self, x):
        # x: (B, L, C)
        x = window_partition(x, self.win_size)
        x = self.proj(x)
        if self.norm_layer:
            x = self.norm(x)
        x = window_reverse(x, self.win_size, self.total_length)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    """
    def __init__(self, total_length=1280, dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.total_length = total_length
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x, mask=None):
        # x: (B, L, C)
        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], -1)
        x = self.norm(x)
        x = self.reduction(x)
        if mask is not None:
            mask0 = mask[:, 0::2]
            mask1 = mask[:, 1::2]
            new_mask = mask0 & mask1  # 둘 다 유효한 위치만 유효하도록
        else:
            new_mask = None
        return x, new_mask

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias=True, attn_drop=0.2, proj_drop=0.2, pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(1, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = relative_coords_w
        if pretrained_window_size > 0:
            relative_coords_table[:] /= (pretrained_window_size - 1)
        else:
            relative_coords_table[:] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table.unsqueeze(1))  # (2*W-1, 1)

        coords_w = torch.arange(self.window_size)
        relative_coords = coords_w[:, None] - coords_w[None, :]
        relative_coords[:, :] += self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape (num_windows*B, N, C)
            mask: attention mask with shape (num_windows*B, N, N) or None
        """
        device = next(self.parameters()).device

        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: (num_windows*B, window_size, window_size)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0.2, attn_drop=0.2, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, C)
            mask: (B, L) boolean tensor indicating valid tokens (True: valid, False: padded)
        Returns:
            x: (B, L, C) updated features
            new_mask: (B, L) updated mask after shift operations
        """
        B, L, C = x.shape

        # 만약 mask가 제공되면, shift 전에 마스크도 동일하게 shift 처리합니다.
        if mask is not None:
            if self.shift_size > 0:
                shifted_mask = torch.roll(mask, shifts=-self.shift_size, dims=1)
                shifted_mask[:, -self.shift_size:] = False
            else:
                shifted_mask = mask
            # window partition: (B, L, 1) -> (B*num_windows, window_size)
            mask_windows = window_partition(shifted_mask.unsqueeze(-1).float(), self.window_size)
            mask_windows = mask_windows.squeeze(-1)  # (B*num_windows, window_size)
            # 각 window 내에서 유효한 토큰이면 1, 아니면 0
            # 어텐션 계산 시, valid 토큰은 0, padded 토큰은 매우 큰 음수(-10000)를 더합니다.
            attn_mask = (1 - (mask_windows.unsqueeze(1) * mask_windows.unsqueeze(2))) * -10000.0
        else:
            attn_mask = None

        shortcut = x
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            shifted_x[:, -self.shift_size:] = 0.
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # (B*num_windows, window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # 어텐션 마스크를 적용하여 계산
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
            x[:, :self.shift_size] = 0.
        else:
            x = shifted_x

        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        if mask is not None:
            if self.shift_size > 0:
                new_mask = torch.roll(shifted_mask, shifts=self.shift_size, dims=1)
                new_mask[:, :self.shift_size] = False
            else:
                new_mask = shifted_mask
        else:
            new_mask = None

        return x, new_mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, L: int):
        flops = 0
        flops += self.dim * L
        nW = L / self.window_size
        flops += nW * self.attn.flops(self.window_size)
        flops += 2 * L * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * L
        return flops

class SwinTransformerV2Layer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio=4., qkv_bias=True, drop=0.2, attn_drop=0.2,
        drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
        pretrained_window_size=0
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x, mask = blk(x, mask)
        return x, mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}, num_heads={self.num_heads}, window_size={self.window_size}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

class Swin1D(nn.Module):
    def __init__(self, max_length=128, window_size=8, dim=1024, feature_dim=1024, num_swin_layers=2, swin_depth=[2, 6], swin_num_heads=[4, 16]):
        super().__init__()

        self.max_length = max_length
        self.window_size = window_size
        self.dim = dim
        self.feature_dim = feature_dim
        self.num_swin_layers = num_swin_layers
        self.swin_depth = swin_depth
        self.swin_num_heads = swin_num_heads

        self.patch_emb = PatchEmbed(total_length=self.max_length, win_size=self.window_size, num_channels=self.dim, emb_size=self.feature_dim, norm_layer=True)

        self.blocks = nn.ModuleList()
        for i in range(self.num_swin_layers):
            self.blocks.append(SwinTransformerV2Layer(dim=self.feature_dim, depth=self.swin_depth[i], num_heads=self.swin_num_heads[i], window_size=self.window_size*(2**i)))
            self.blocks.append(PatchMerging(total_length=self.max_length//(2**i), dim=self.feature_dim))

        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.feature_dim//2),
        )
        # 최종 분류를 위한 classifier layer (예: 2 클래스 분류)
        self.classifier = nn.Linear(self.feature_dim//2, 2)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, dim)
            mask: (B, L) boolean tensor indicating valid tokens
        """
        # Patch embedding (mask는 그대로 유지)
        x = self.patch_emb(x)
        # blocks 순차적으로 진행 (mask도 함께 전달)
        for layer in self.blocks:
            x, mask = layer(x, mask)
        # Masked average pooling: valid한 토큰만 평균 내기
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = x * mask_expanded
            valid_counts = mask_expanded.sum(dim=1).clamp(min=1)
            x = x.sum(dim=1) / valid_counts
        else:
            x = x.mean(dim=1)
        x = self.head(x)
        x = self.classifier(x)
        return x
