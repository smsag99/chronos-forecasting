import pytest
import torch
from transformers import BartConfig
from chronos.chronos_bolt_bart import (
    ChronosBartConfig,
    ChronosBartForForecasting,
    ChronosBartPipeline,
    InstanceNorm,
    Patch
)

@pytest.fixture
def model_config():
    """Create a basic BART config with Chronos settings"""
    base_config = BartConfig(
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
    )
    
    chronos_config = {
        "context_length": 128,
        "prediction_length": 24,
        "input_patch_size": 16,
        "input_patch_stride": 8,
        "quantiles": [0.1, 0.5, 0.9],
        "use_reg_token": False
    }
    
    base_config.chronos_config = chronos_config
    return base_config

@pytest.fixture
def sample_data():
    """Create sample time series data"""
    batch_size = 4
    context_length = 96
    
    # Create synthetic time series with seasonal pattern
    t = torch.linspace(0, 6*torch.pi, context_length)
    base_series = torch.sin(t) + 0.1 * torch.randn_like(t)
    
    # Create batch with variations
    context = torch.stack([
        base_series + 0.5 * torch.randn_like(base_series)
        for _ in range(batch_size)
    ])
    
    return context

def test_instance_norm():
    """Test instance normalization layer"""
    instance_norm = InstanceNorm()
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Test forward pass
    normalized, (loc, scale) = instance_norm(x)
    assert normalized.shape == x.shape
    assert loc.shape == (2, 1)
    assert scale.shape == (2, 1)
    
    # Test inverse transform
    reconstructed = instance_norm.inverse(normalized, (loc, scale))
    assert torch.allclose(reconstructed, x, rtol=1e-5)

def test_patch_layer():
    """Test patching functionality"""
    patch_size = 4
    patch_stride = 2
    patch_layer = Patch(patch_size=patch_size, patch_stride=patch_stride)
    
    # Test with perfect divisible input
    x = torch.randn(2, 16)
    patched = patch_layer(x)
    assert patched.shape[0] == 2
    assert patched.shape[-1] == patch_size
    
    # Test with non-divisible input
    x = torch.randn(2, 7)
    patched = patch_layer(x)
    assert patched.shape[0] == 2
    assert patched.shape[-1] == patch_size

def test_model_initialization(model_config):
    """Test model creation and basic properties"""
    model = ChronosBartForForecasting(model_config)
    
    assert isinstance(model, ChronosBartForForecasting)
    assert hasattr(model, 'bart')
    assert hasattr(model, 'patch')
    assert hasattr(model, 'instance_norm')
    
    # Check quantile setup
    assert model.num_quantiles == len(model_config.chronos_config["quantiles"])
    assert model.quantiles.shape[0] == len(model_config.chronos_config["quantiles"])

def test_model_forward(model_config, sample_data):
    """Test model forward pass"""
    model = ChronosBartForForecasting(model_config)
    
    # Test without target (inference mode)
    outputs = model(sample_data)
    assert outputs.quantile_preds is not None
    assert outputs.loss is None
    
    # Check output shapes
    batch_size = sample_data.shape[0]
    pred_length = model_config.chronos_config["prediction_length"]
    num_quantiles = len(model_config.chronos_config["quantiles"])
    
    assert outputs.quantile_preds.shape == (batch_size, num_quantiles, pred_length)
    
    # Test with target
    target = torch.randn(batch_size, pred_length)
    outputs_with_target = model(sample_data, target=target)
    assert outputs_with_target.loss is not None
    assert isinstance(outputs_with_target.loss, torch.Tensor)

def test_pipeline_prediction(model_config, sample_data):
    """Test pipeline forecasting functionality"""
    model = ChronosBartForForecasting(model_config)
    pipeline = ChronosBartPipeline(model)
    
    # Test basic prediction
    predictions = pipeline.predict(sample_data)
    assert predictions.shape[1] == len(model_config.chronos_config["quantiles"])
    assert predictions.shape[2] == model_config.chronos_config["prediction_length"]
    
    # Test quantile prediction
    quantiles, mean = pipeline.predict_quantiles(
        sample_data,
        quantile_levels=[0.1, 0.5, 0.9]
    )
    assert quantiles.shape[-1] == 3  # number of requested quantiles
    assert mean.shape == (sample_data.shape[0], model_config.chronos_config["prediction_length"])

def test_longer_prediction(model_config, sample_data):
    """Test prediction beyond training length"""
    model = ChronosBartForForecasting(model_config)
    pipeline = ChronosBartPipeline(model)
    
    longer_length = model_config.chronos_config["prediction_length"] * 2
    
    # Should warn but not fail
    with pytest.warns(UserWarning):
        predictions = pipeline.predict(
            sample_data,
            prediction_length=longer_length,
            limit_prediction_length=False
        )
    assert predictions.shape[2] == longer_length
    
    # Should fail when limit_prediction_length=True
    with pytest.raises(ValueError):
        pipeline.predict(
            sample_data,
            prediction_length=longer_length,
            limit_prediction_length=True
        )

if __name__ == "__main__":
    pytest.main([__file__])