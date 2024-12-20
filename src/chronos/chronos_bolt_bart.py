import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from venv import logger
import torch
import torch.nn as nn
from transformers import BartPreTrainedModel, BartConfig, AutoConfig
from transformers import (
    BartPreTrainedModel,
    BartConfig,
    BartModel,
    PreTrainedModel
)
from transformers.utils import ModelOutput

from chronos.base import BaseChronosPipeline, ForecastType


@dataclass
class ChronosBartConfig:
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
    
class ChronosBartForForecasting(BartPreTrainedModel):
    """BART-based model for time series forecasting"""
    def __init__(self, config: BartConfig):
        super().__init__(config)
        
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"
        
        self.model_dim = config.d_model
        self.chronos_config = ChronosBartConfig(**config.chronos_config)
        
        # Initialize BART model
        self.bart = BartModel(config)
        
        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            in_dim=self.chronos_config.input_patch_size * 2,
            h_dim=config.encoder_ffn_dim,
            out_dim=config.d_model,
            dropout_p=config.dropout
        )
        
        # Patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride
        )
        
        # Instance normalization
        self.instance_norm = InstanceNorm()
        
        # Prepare quantiles
        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles, dtype=torch.float32)
        self.register_buffer("quantiles", quantiles, persistent=False)
        
        # Output projection
        self.output_patch_embedding = ResidualBlock(
            in_dim=config.d_model,
            h_dim=config.decoder_ffn_dim,
            out_dim=self.num_quantiles * self.chronos_config.prediction_length,
            dropout_p=config.dropout
        )
        
        self.post_init()

    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> ChronosBartOutput:
        
        mask = (
            mask.to(context.dtype)
            if mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )
        
        batch_size, _ = context.shape
        
        if context.shape[-1] > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length:]
            mask = mask[..., -self.chronos_config.context_length:]
            
        # Apply instance normalization
        context, loc_scale = self.instance_norm(context)
        context = context.to(self.dtype)
        mask = mask.to(self.dtype)
        
        # Apply patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)
        
        # Create attention mask
        attention_mask = (patched_mask.sum(dim=-1) > 0)
        
        # Get embeddings
        input_embeds = self.input_patch_embedding(patched_context)
        
        if self.chronos_config.use_reg_token:
            reg_embeds = self.bart.shared(
                torch.full((batch_size, 1), 1, device=input_embeds.device)
            )
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(reg_embeds[:,:,0])],
                dim=-1
            )
            
        # Pass through BART
        outputs = self.bart(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get predictions
        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        
        sequence_output = outputs.last_hidden_state[:,-1:,:]
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )
        
        loss = None
        if target is not None:
            # Normalize target
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            
            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0
            
            # Pad if needed
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)],
                    dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)],
                    dim=-1
                )
                
            # Compute quantile loss
            loss = (
                2
                * torch.abs(
                    (target - quantile_preds)
                    * (
                        (target <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2)
            loss = loss.sum(dim=-1)
            loss = loss.mean()
            
        # Unscale predictions
        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)
        
        return ChronosBartOutput(
            loss=loss,
            quantile_preds=quantile_preds,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions
        )
    
    def _init_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.decoder_layers
        self.decoder = self.bart.decoder

    def decode(
        self,
        input_embeds,
        attention_mask,
        hidden_states,
        output_attentions=False,
    ):
        """
        Parameters
        ----------
        input_embeds: torch.Tensor
            Patched and embedded inputs. Shape (batch_size, patched_context_length, d_model)
        attention_mask: torch.Tensor
            Attention mask for the patched context. Shape (batch_size, patched_context_length)
        hidden_states: torch.Tensor
            Hidden states returned by the encoder. Shape (batch_size, patched_context_length, d_model)

        Returns
        -------
        last_hidden_state
            Last hidden state returned by the decoder, shape (batch_size, 1, d_model)
        """
        batch_size = input_embeds.shape[0]
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            device=input_embeds.device,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return decoder_outputs.last_hidden_state


class ChronosBartPipeline(BaseChronosPipeline):
    forecast_type: ForecastType = ForecastType.QUANTILES
    default_context_length: int = 2048

    def __init__(self, model: ChronosBartForForecasting):
        super().__init__(inner_model=model)
        self.model = model

    @property
    def quantiles(self) -> List[float]:
        return self.model.config.chronos_config["quantiles"]

    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        limit_prediction_length: bool = False,
    ) -> torch.Tensor:
        """
        Get forecasts for the given time series.

        Parameters
        ----------
        context
            Input series. Can be a 1D tensor, list of 1D tensors, or 2D tensor 
            with batch as first dimension
        prediction_length
            Time steps to predict. Defaults to model's built-in prediction length
        limit_prediction_length
            If True, fail if prediction_length exceeds model's training length

        Returns
        -------
        torch.Tensor
            Forecasts of shape (batch_size, num_quantiles, prediction_length)
        """
        context_tensor = self._prepare_and_validate_context(context=context)

        model_prediction_length = self.model.config.chronos_config["prediction_length"]
        if prediction_length is None:
            prediction_length = model_prediction_length

        if prediction_length > model_prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {model_prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        # Handle long context
        if context_tensor.shape[-1] > self.model.config.chronos_config["context_length"]:
            context_tensor = context_tensor[..., -self.model.config.chronos_config["context_length"]:]

        while remaining > 0:
            with torch.no_grad():
                prediction = self.model(
                    context=context_tensor.to(
                        device=self.model.device,
                        dtype=torch.float32,
                    ),
                ).quantile_preds.to(context_tensor)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            central_idx = torch.abs(torch.tensor(self.quantiles) - 0.5).argmin()
            central_prediction = prediction[:, central_idx]
            context_tensor = torch.cat([context_tensor, central_prediction], dim=-1)

        return torch.cat(predictions, dim=-1)[..., :prediction_length]

    def predict_quantiles(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get quantile and mean forecasts for given time series.

        Parameters same as predict() plus:
        quantile_levels: List[float]
            Quantile levels to compute, default [0.1, 0.2, ..., 0.9]

        Returns
        -------
        quantiles: torch.Tensor 
            Shape (batch_size, prediction_length, num_quantiles)
        mean: torch.Tensor
            Shape (batch_size, prediction_length)
        """
        predictions = (
            self.predict(context, prediction_length=prediction_length, **predict_kwargs)
            .detach()
            .swapaxes(1, 2)
        )

        training_quantile_levels = self.quantiles

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            quantiles = predictions[
                ..., [training_quantile_levels.index(q) for q in quantile_levels]
            ]
        else:
            if min(quantile_levels) < min(training_quantile_levels) or max(
                quantile_levels
            ) > max(training_quantile_levels):
                logger.warning(
                    f"\tRequested quantiles ({quantile_levels}) are outside the range of "
                    f"training quantiles ({training_quantile_levels}). "
                    "Predictions will be capped at training quantile range."
                )

            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions,
                q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
                dim=-1,
            ).permute(1, 2, 0)

        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load the model from local path or HuggingFace Hub.
        """
        config = AutoConfig.from_pretrained(*args, **kwargs)
        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        architecture = config.architectures[0]
        class_ = globals().get(architecture)

        if class_ is None:
            logger.warning(
                f"Unknown architecture: {architecture}, defaulting to ChronosBartForForecasting"
            )
            class_ = ChronosBartForForecasting

        model = class_.from_pretrained(*args, **kwargs)
        return cls(model=model)