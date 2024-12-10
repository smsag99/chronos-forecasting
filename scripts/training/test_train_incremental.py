import os
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import torch
import yaml
import logging
from gluonts.dataset.arrow import ArrowWriter

# Import data generation code
from data_generation import (
    TSMixup, 
    TSMixupConfig,
    generate_kernel_synth_ts,
    load_chronos_datasets
)
from train_incremental import incremental_training
from chronos import ChronosConfig, ChronosTokenizer
from chronos.chronos import ChronosModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(tmp_dir: Path, n_synthetic: int = 5, n_mixup: int = 5):
    """Generate test data using both KernelSynth and TSMixup"""
    logger.info("Loading source datasets for TSMixup...")
    series_list = load_chronos_datasets(max_zero_or_nan=0.9)
    
    # Generate KernelSynth data
    logger.info(f"Generating {n_synthetic} KernelSynth time series...")
    synthetic_data = [
        generate_kernel_synth_ts()
        for _ in range(n_synthetic)
    ]
    
    # Generate TSMixup augmentations
    logger.info(f"Generating {n_mixup} TSMixup augmentations...")
    mixup = TSMixup(TSMixupConfig())
    mixup_data = [
        mixup.generate_single_mix(series_list)
        for _ in range(n_mixup)
    ]
    
    # Save both types of data
    kernelsynth_path = tmp_dir / "kernelsynth_data.arrow"
    tsmixup_path = tmp_dir / "tsmixup_data.arrow"
    
    ArrowWriter(compression="lz4").write_to_file(
        synthetic_data,
        kernelsynth_path
    )
    ArrowWriter(compression="lz4").write_to_file(
        mixup_data, 
        tsmixup_path
    )
    
    logger.info(f"Saved test data to {tmp_dir}")
    return kernelsynth_path, tsmixup_path

def create_test_config(data_paths):
    """Create a minimal test configuration"""
    config = {
        "training_data_paths": data_paths,
        "probability": [0.5, 0.5],  # Equal probability for both data sources
        "context_length": 64,
        "prediction_length": 16,
        "min_past": 32,
        "max_steps": 20,  # Small number of steps for testing
        "save_steps": 5,
        "log_steps": 2,
        "per_device_train_batch_size": 2,
        "learning_rate": 1e-4,
        "gradient_accumulation_steps": 2,
        "model_id": "google/t5-efficient-tiny",
        "model_type": "seq2seq",
        "tokenizer_class": "MeanScaleUniformBins",
        "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0},
        "n_tokens": 4096,
        "n_special_tokens": 2,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "use_eos_token": True,
        "num_samples": 10,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "tf32": False,  # Disable for testing
        "torch_compile": False,  # Disable for testing
    }
    return config

def test_training_pipeline():
    """Run a complete test of the training pipeline"""
    try:
        # Create temporary directory for test artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            logger.info(f"Created temporary directory: {tmp_dir}")
            
            # Generate test data using both methods
            logger.info("Generating test data...")
            kernelsynth_path, tsmixup_path = generate_test_data(tmp_dir)
            
            # Create and save test config with both data sources
            config = create_test_config([str(kernelsynth_path), str(tsmixup_path)])
            config['output_dir'] = str(tmp_dir / "output")
            config_path = tmp_dir / "test_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            logger.info(f"Saved test config to {config_path}")
            
            # Run first batch
            logger.info("Testing first batch training...")
            incremental_training(
                config_path=str(config_path),
                checkpoint_path=None,
                batch_size=5,
                batch_start=0,
                device_map="cpu"
            )
            
            first_batch_path = (
                tmp_dir 
                / "output" 
                / "batch_0" 
                / "checkpoint-final-batch-0"
            )
            assert first_batch_path.exists(), f"First checkpoint not created at {first_batch_path}"
            logger.info(f"First batch checkpoint created at {first_batch_path}")
            
            
            # Run second batch from checkpoint
            logger.info("Testing second batch training from checkpoint...")
            # Run second batch using correct checkpoint path
            incremental_training(
                config_path=str(config_path),
                checkpoint_path=str(first_batch_path),
                batch_size=5,
                batch_start=1,
                device_map="cpu"
            )
            
            # Check second checkpoint with correct path
            second_batch_path = (
                tmp_dir 
                / "output" 
                / "batch_1" 
                / "checkpoint-final-batch-1"
            )
            assert second_batch_path.exists(), f"Second checkpoint not created at {second_batch_path}"
            logger.info(f"Second batch checkpoint created at {second_batch_path}")
            
            # Load final model and try a prediction
            logger.info("Testing model prediction...")
            from transformers import AutoConfig, AutoModelForSeq2SeqLM
            
            config = AutoConfig.from_pretrained(second_batch_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(second_batch_path)
            chronos_config = ChronosConfig(**config.chronos_config)
            
            # Create a simple input
            test_input = torch.tensor([[1.0] * 32], dtype=torch.float32)
            tokenizer = chronos_config.create_tokenizer()
            token_ids, attention_mask, scale = tokenizer.context_input_transform(test_input)
            
            # Generate prediction
            with torch.no_grad():
                output = model.generate(
                    input_ids=token_ids,
                    attention_mask=attention_mask,
                    max_length=chronos_config.prediction_length + 1,
                    num_return_sequences=1
                )
            
            assert output.shape[1] > 0, "Model failed to generate output"
            logger.info("Model prediction successful")
            
            # Try making a prediction on real test data
            logger.info("Testing prediction on generated data...")
            # Load one series from generated data
            from gluonts.dataset.common import FileDataset
            test_dataset = FileDataset(kernelsynth_path)
            test_series = next(iter(test_dataset))
            test_input = torch.tensor([test_series["target"][:32]], dtype=torch.float32)
            
            token_ids, attention_mask, scale = tokenizer.context_input_transform(test_input)
            with torch.no_grad():
                output = model.generate(
                    input_ids=token_ids,
                    attention_mask=attention_mask,
                    max_length=chronos_config.prediction_length + 1,
                    num_return_sequences=1
                )
                
            assert output.shape[1] > 0, "Model failed to generate output for test data"
            logger.info("Model prediction on test data successful")
            
            logger.info("All tests passed successfully!")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_training_pipeline()