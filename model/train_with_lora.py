"""
LoRA Fine-tuning Script for LCF Group Funding Recommendation System

Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA to fine-tune
a transformer model for business funding recommendations.
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import datasets
from datasets import Dataset
import boto3
from botocore.exceptions import ClientError
import wandb
from loguru import logger

# Configure logging
logger.add("logs/training.log", rotation="10 MB", level="INFO")


class FundingDataset(Dataset):
    """Custom dataset for funding recommendation data"""
    
    def __init__(self, texts: List[str], features: List[Dict], labels: List[List[int]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        feature_dict = self.features[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Flatten encoding
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }
        
        # Add numerical features
        for key, value in feature_dict.items():
            if isinstance(value, (int, float)):
                item[f'feature_{key}'] = torch.tensor(value, dtype=torch.float32)
        
        return item


class FundingModel:
    """Main model class for funding recommendations"""
    
    def __init__(self, config_path: str = "model/config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loaded configuration from: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _load_or_create_tokenizer(self):
        """Load or create tokenizer"""
        model_name = self.config['model']['base_model']
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def _load_base_model(self):
        """Load base transformer model"""
        model_name = self.config['model']['base_model']
        num_labels = self.config['model']['classification']['num_labels']
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="multi_label_classification"
            )
            logger.info(f"Loaded base model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
    
    def _setup_lora_config(self):
        """Setup LoRA configuration"""
        lora_config = self.config['model']['lora_config']
        
        return LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type=TaskType.SEQ_CLS
        )
    
    def _create_peft_model(self):
        """Create PEFT model with LoRA"""
        lora_config = self._setup_lora_config()
        
        try:
            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.print_trainable_parameters()
            logger.info("Created PEFT model with LoRA")
        except Exception as e:
            logger.error(f"Error creating PEFT model: {e}")
            raise
    
    def _prepare_data(self, data_path: str) -> Tuple[List[str], List[Dict], List[List[int]]]:
        """Prepare training data"""
        try:
            # Load preprocessed data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            texts = []
            features = []
            labels = []
            
            for item in data:
                # Extract text (business description)
                text = item.get('cleaned_description', '')
                if not text:
                    continue
                
                # Extract features
                feature_dict = item.get('features', {})
                
                # Create labels (simplified - in production, use actual labels)
                # For demo: create synthetic labels based on features
                risk_score = self._calculate_risk_score(feature_dict)
                funding_score = self._calculate_funding_score(feature_dict)
                
                # Convert to multi-label format
                label = [0] * 6  # 3 risk levels + 3 funding levels
                label[risk_score] = 1
                label[funding_score + 3] = 1
                
                texts.append(text)
                features.append(feature_dict)
                labels.append(label)
            
            logger.info(f"Prepared {len(texts)} training samples")
            return texts, features, labels
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _calculate_risk_score(self, features: Dict) -> int:
        """Calculate risk score based on features (simplified)"""
        revenue = features.get('revenue', 0)
        employee_count = features.get('employee_count', 0)
        years_active = features.get('years_active', 0)
        
        # Simple risk calculation
        risk_factors = 0
        if revenue < 100000:
            risk_factors += 1
        if employee_count < 10:
            risk_factors += 1
        if years_active < 2:
            risk_factors += 1
        
        if risk_factors >= 2:
            return 2  # High risk
        elif risk_factors >= 1:
            return 1  # Medium risk
        else:
            return 0  # Low risk
    
    def _calculate_funding_score(self, features: Dict) -> int:
        """Calculate funding recommendation based on features (simplified)"""
        revenue = features.get('revenue', 0)
        revenue_per_employee = features.get('revenue_per_employee', 0)
        
        # Simple funding calculation
        if revenue > 1000000 and revenue_per_employee > 50000:
            return 0  # Yes
        elif revenue > 500000:
            return 2  # Maybe
        else:
            return 1  # No
    
    def _create_datasets(self, texts: List[str], features: List[Dict], labels: List[List[int]]) -> Tuple[Dataset, Dataset]:
        """Create training and validation datasets"""
        # Split data
        train_texts, val_texts, train_features, val_features, train_labels, val_labels = train_test_split(
            texts, features, labels, 
            test_size=self.config['model']['evaluation']['validation_split'],
            random_state=self.config['model']['evaluation']['random_state']
        )
        
        # Create datasets
        train_dataset = FundingDataset(
            train_texts, train_features, train_labels, 
            self.tokenizer, 
            self.config['model']['data']['max_length']
        )
        
        val_dataset = FundingDataset(
            val_texts, val_features, val_labels,
            self.tokenizer,
            self.config['model']['data']['max_length']
        )
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        return train_dataset, val_dataset
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics for each label
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        
        # Calculate ROC AUC for each label
        roc_auc_scores = []
        for i in range(labels.shape[1]):
            try:
                roc_auc = roc_auc_score(labels[:, i], predictions[:, i])
                roc_auc_scores.append(roc_auc)
            except ValueError:
                roc_auc_scores.append(0.0)
        
        avg_roc_auc = np.mean(roc_auc_scores)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': avg_roc_auc
        }
    
    def train(self, data_path: str, output_dir: str = "models/lora_finetuned"):
        """Train the model with LoRA fine-tuning"""
        try:
            logger.info("Starting model training...")
            
            # Initialize components
            self._load_or_create_tokenizer()
            self._load_base_model()
            self._create_peft_model()
            
            # Prepare data
            texts, features, labels = self._prepare_data(data_path)
            train_dataset, val_dataset = self._create_datasets(texts, features, labels)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=self.config['model']['training']['learning_rate'],
                per_device_train_batch_size=self.config['model']['training']['batch_size'],
                per_device_eval_batch_size=self.config['model']['training']['batch_size'],
                num_train_epochs=self.config['model']['training']['epochs'],
                weight_decay=self.config['model']['training']['weight_decay'],
                logging_dir=f"{output_dir}/logs",
                logging_steps=self.config['model']['training']['logging_steps'],
                evaluation_strategy="steps",
                eval_steps=self.config['model']['training']['eval_steps'],
                save_steps=self.config['model']['training']['save_steps'],
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                warmup_steps=self.config['model']['training']['warmup_steps'],
                gradient_accumulation_steps=self.config['model']['training']['gradient_accumulation_steps'],
                max_grad_norm=self.config['model']['training']['max_grad_norm'],
                report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self._compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Train model
            logger.info("Training model...")
            trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Save configuration
            config_save_path = os.path.join(output_dir, "model_config.json")
            with open(config_save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Model saved to: {output_dir}")
            
            # Evaluate final model
            logger.info("Evaluating final model...")
            eval_results = trainer.evaluate()
            logger.info(f"Final evaluation results: {eval_results}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def save_to_s3(self, local_path: str, s3_bucket: str = None, s3_prefix: str = None):
        """Save model to S3"""
        try:
            if not s3_bucket:
                s3_bucket = self.config['model']['aws']['s3_bucket']
            if not s3_prefix:
                s3_prefix = self.config['model']['aws']['s3_prefix']
            
            s3_client = boto3.client('s3')
            
            # Upload model files
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    s3_key = os.path.join(s3_prefix, os.path.relpath(local_file, local_path))
                    
                    s3_client.upload_file(local_file, s3_bucket, s3_key)
                    logger.info(f"Uploaded {local_file} to s3://{s3_bucket}/{s3_key}")
            
            logger.info(f"Model successfully uploaded to S3: s3://{s3_bucket}/{s3_prefix}")
            
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {e}")
            raise


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train LoRA model for funding recommendations")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed data")
    parser.add_argument("--config_path", type=str, default="model/config/model_config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="models/lora_finetuned", help="Output directory")
    parser.add_argument("--upload_s3", action="store_true", help="Upload model to S3 after training")
    
    args = parser.parse_args()
    
    try:
        # Initialize model
        model = FundingModel(args.config_path)
        
        # Train model
        output_path = model.train(args.data_path, args.output_dir)
        
        # Upload to S3 if requested
        if args.upload_s3:
            model.save_to_s3(output_path)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 