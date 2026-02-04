"""
Training script for CXR-LLaVA fine-tuning.
"""

import os
import json
import argparse
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class LlavaDataCollator:
    """Data collator for LLaVA training."""
    
    def __init__(self, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch):
        images = []
        texts = []
        
        for ex in batch:
            images.append(Image.open(ex["image"]).convert("RGB"))
            
            user = ex["conversations"][0]["value"]
            assistant = ex["conversations"][1]["value"]
            texts.append(f"USER: {user}\nASSISTANT: {assistant}")
        
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def main():
    parser = argparse.ArgumentParser(description="Train CXR-LLaVA model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for model upload"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        login(args.hf_token)
    
    # Load processor
    print("Loading processor...")
    processor = LlavaProcessor.from_pretrained(config["model"]["base_model"])
    
    # Load base model
    print("Loading base model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=getattr(torch, config["model"]["torch_dtype"]),
        device_map=config["model"]["device_map"]
    )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"Loading dataset from {args.data}...")
    with open(args.data) as f:
        raw_data = json.load(f)
    
    dataset = Dataset.from_list(raw_data)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        fp16=config["training"]["fp16"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        remove_unused_columns=config["training"]["remove_unused_columns"],
        report_to=config["training"]["report_to"],
        push_to_hub=config["hub"]["push_to_hub"],
        hub_model_id=config["hub"]["hub_model_id"] if config["hub"]["push_to_hub"] else None,
        hub_strategy=config["hub"]["hub_strategy"] if config["hub"]["push_to_hub"] else None,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=LlavaDataCollator(
            processor, 
            max_length=config["data"]["max_length"]
        ),
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Push to hub
    if config["hub"]["push_to_hub"]:
        print("Pushing model to HuggingFace Hub...")
        trainer.push_to_hub()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
