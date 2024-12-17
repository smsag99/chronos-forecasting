from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import BartPreTrainedModel, BartConfig
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartDecoder,
    LayerNorm,
    shift_tokens_right,
)
from transformers.utils import ModelOutput


@dataclass
class ChornosBartConfig:
    """configurations class for chronos-bart architecture"""
    context_lenght: int
    prediction_length: int
    nput_patch_size: int
    input_patch_stride: int
    quantiles: List[float]
    use_reg_token: bool = False

class ChronosBartOutput:
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None

class Patch(nn.Module):
    """time series patching layer"""
    def __init__(self, patch_size, patch_stride) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        
        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, 
                fill_value=torch.nan,
                dtype=x.dtype,
                device=x.device
            )
            x = torch.concat((padding, x), dim=-1)
            
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x

class InstanceNorm(nn.Module):
    """Instance normalization layer"""
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), 
                nan=1.0
            )
            scale = torch.where(scale == 0, torch.abs(loc) + self.eps, scale)
        else:
            loc, scale = loc_scale

        return (x - loc) / scale, (loc, scale)

    def inverse(
        self, 
        x: torch.Tensor,
        loc_scale: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loc, scale = loc_scale
        return x * scale + loc


class ResidualBlock(nn.Module):
    """Residual block with optional layer normalization"""
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = nn.GELU()
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        
        out = out + res
        
        if self.use_layer_norm:
            return self.layer_norm(out)
        return out
    
    