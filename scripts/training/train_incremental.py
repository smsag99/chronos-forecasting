import argparse
import logging
import os
import yaml
from pathlib import Path
from typing import Optional, Dict

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from chronos import ChronosConfig, ChronosTokenizer
from chronos.chronos import ChronosModel
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic
from train import (
    ChronosDataset,
    Filter,
    has_enough_observations,
)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def load_yaml_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_for_incremental(config: Dict, batch_start: int, checkpoint_path: str) -> Dict:
    """Update configuration for incremental training"""
    # Create a copy to avoid modifying the original
    config = config.copy()
    
    # Update training data path to only include one path
    if isinstance(config['training_data_paths'], list):
        config['training_data_paths'] = [config['training_data_paths'][0]]
    if isinstance(config['probability'], list):
        config['probability'] = [1.0]
        
    # Update model path to use checkpoint
    if checkpoint_path:
        config['model_id'] = checkpoint_path
        
    # Adjust batch size and other parameters for incremental training
    config['per_device_train_batch_size'] = min(
        config.get('per_device_train_batch_size', 32),
        8  # Smaller default batch size for incremental training
    )
    
    # Add batch information to output directory
    config['output_dir'] = str(Path(config['output_dir']) / f"batch_{batch_start}")
    
    return config

def load_checkpoint(checkpoint_path: str, device_map: str = "auto"):
    """Load model from checkpoint"""
    config = AutoConfig.from_pretrained(checkpoint_path)
    chronos_config = ChronosConfig(**config.chronos_config)

    if chronos_config.model_type == "seq2seq":
        inner_model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path,
            device_map=device_map,
        )
    else:
        inner_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, 
            device_map=device_map,
        )
    
    return ChronosModel(config=chronos_config, model=inner_model), chronos_config

def load_data_batch(data_path: str, batch_start: int, batch_size: int, min_past: int, prediction_length: int, max_missing_prop: float = 0.9):
    """Load and filter a batch of data"""
    dataset = FileDataset(path=Path(data_path), freq="h")
    filtered_dataset = Filter(
        lambda x: has_enough_observations(
            x,
            min_length=min_past + prediction_length,
            max_missing_prop=max_missing_prop,
        ),
        dataset,
    )
    
    # Get the specified batch
    all_data = list(filtered_dataset)
    start_idx = batch_start * batch_size
    end_idx = min((batch_start + 1) * batch_size, len(all_data))
    
    return all_data[start_idx:end_idx]

def incremental_training(
    config_path: str,
    checkpoint_path: Optional[str],
    batch_size: int = 1000,
    batch_start: int = 0,
    device_map: str = "auto",
):
    """Incrementally train a Chronos model on batches of data using YAML config"""
    
    # Load and update config
    config = load_yaml_config(config_path)
    config = update_config_for_incremental(config, batch_start, checkpoint_path)
    
    # Load model and config from checkpoint if provided, otherwise use config
    if checkpoint_path:
        model, chronos_config = load_checkpoint(checkpoint_path, device_map)
    else:
        # Initialize new model from config
        chronos_config = ChronosConfig(
            tokenizer_class=config['tokenizer_class'],
            tokenizer_kwargs=config['tokenizer_kwargs'],
            context_length=config['context_length'],
            prediction_length=config['prediction_length'],
            n_tokens=config['n_tokens'],
            n_special_tokens=config['n_special_tokens'],
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            use_eos_token=config['use_eos_token'],
            model_type=config['model_type'],
            num_samples=config['num_samples'],
            temperature=config['temperature'],
            top_k=config['top_k'],
            top_p=config['top_p'],
        )
        
        # Create model config first
        model_config = AutoConfig.from_pretrained(config['model_id'])
        # Add chronos config to model config
        model_config.chronos_config = chronos_config.__dict__
        
        # Initialize model with proper config
        if config['model_type'] == "seq2seq":
            inner_model = AutoModelForSeq2SeqLM.from_pretrained(
                config['model_id'], 
                config=model_config
            )
        else:
            inner_model = AutoModelForCausalLM.from_pretrained(
                config['model_id'], 
                config=model_config
            )
        model = ChronosModel(config=chronos_config, model=inner_model)
    
    # Load data batch
    train_data = load_data_batch(
        data_path=config['training_data_paths'][0],
        batch_start=batch_start,
        batch_size=batch_size,
        min_past=config['min_past'],
        prediction_length=config['prediction_length'],
    )
    
    if not train_data:
        logger.warning(f"No data found in batch {batch_start}")
        return

    logger.info(f"Training on {len(train_data)} series from batch {batch_start}")
    
    # Create dataset
    train_dataset = ChronosDataset(
        datasets=[train_data],
        probabilities=[1.0],
        tokenizer=chronos_config.create_tokenizer(),
        context_length=chronos_config.context_length,
        prediction_length=chronos_config.prediction_length,
        min_past=config['min_past'],
        model_type=chronos_config.model_type,
        mode="training",
    )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        max_steps=config.get('max_steps', 1000),
        save_steps=config.get('save_steps', 500),
        logging_steps=config.get('log_steps', 100),
        dataloader_num_workers=config.get('dataloader_num_workers', 1),
        tf32=config.get('tf32', True),
        torch_compile=config.get('torch_compile', True),
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model.model,  # Always use the inner model
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()
    
    # Save final checkpoint
    output_path = f"{config['output_dir']}/checkpoint-final-batch-{batch_start}"
    trainer.save_model(output_path)
    logger.info(f"Saved final checkpoint to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                      help="Path to YAML config file")
    parser.add_argument("--checkpoint_path", type=str,
                      help="Path to previous checkpoint (optional)")
    parser.add_argument("--batch_size", type=int, default=1000,
                      help="Number of series per batch")
    parser.add_argument("--batch_start", type=int, default=0,
                      help="Starting batch index") 
    parser.add_argument("--device_map", type=str, default="auto",
                      help="Device mapping strategy")
    
    args = parser.parse_args()
    
    incremental_training(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        batch_start=args.batch_start,
        device_map=args.device_map,
    )