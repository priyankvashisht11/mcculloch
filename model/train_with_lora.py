"""
LoRA Fine-tuning Script for LCF Group Funding Recommendation System

Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA to fine-tune
a transformer model for business funding recommendations.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

torch_device = "auto"
if torch.cuda.is_available():
    torch_device = "cuda"
# Force everything to CPU
import json
import yaml
import argparse
import boto3
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from botocore.exceptions import ClientError

# Transformers imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback

# PEFT imports
from peft import LoraConfig, TaskType, get_peft_model

# Logging
from loguru import logger

# Configure logging
logger.add("logs/training.log", rotation="10 MB", level="INFO")

print("TrainingArguments source:", TrainingArguments.__module__)

class TextGenDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, responses, tokenizer, max_length=1024):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        # Concatenate prompt and response for causal LM
        full_input = prompt + response
        encoding = self.tokenizer(
            full_input,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # For CausalLM, labels are the same as input_ids (with padding tokens masked)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding tokens for loss
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class FundingModel:
    """Main model class for funding recommendations"""
    
    def __init__(self, config_path: str = "model/config/model_config.yaml"):
        self.config = self._load_config(config_path)
        # Set device to CUDA if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
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
            if self.device.type == "cuda":
                try:
                    from transformers.utils.quantization_config import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        num_labels=num_labels,
                        problem_type="multi_label_classification",
                        quantization_config=bnb_config,
                        device_map="cpu",
                        torch_dtype=torch.float16
                    )
                    self.model = self.model.to(self.device)
                    logger.info(f"Loaded base model with 8-bit quantization: {model_name}")
                except Exception as quant_error:
                    logger.warning(f"Quantization failed, trying without: {quant_error}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        num_labels=num_labels,
                        problem_type="multi_label_classification",
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    self.model = self.model.to(self.device)
                    logger.info(f"Loaded base model without quantization: {model_name}")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    problem_type="multi_label_classification",
                    device_map={"": "cpu"},
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
                logger.info(f"Loaded base model on CPU: {model_name}")
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
        if self.model is None:
            raise ValueError("Base model must be loaded before creating PEFT model")
            
        lora_config = self._setup_lora_config()
        
        try:
            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.print_trainable_parameters()
            logger.info("Created PEFT model with LoRA")
        except Exception as e:
            logger.error(f"Error creating PEFT model: {e}")
            raise
    
    def _prepare_data(self, data_path: str):
        prompts = []
        responses = []
        with open(data_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                response = obj.get("response", "")
                if prompt and response:
                    prompts.append(prompt)
                    responses.append(response)
        print(f"Loaded {len(prompts)} samples")
        return prompts, responses
    
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
    
    def _create_datasets(self, prompts: List[str], responses: List[str]) -> Tuple[TextGenDataset, TextGenDataset]:
        # Split data
        train_prompts, val_prompts, train_responses, val_responses = train_test_split(
            prompts, responses,
            test_size=self.config['model']['evaluation']['validation_split'],
            random_state=self.config['model']['evaluation']['random_state']
        )
        # Create datasets
        train_dataset = TextGenDataset(train_prompts, train_responses, self.tokenizer, self.config['model']['data']['max_length'])
        val_dataset = TextGenDataset(val_prompts, val_responses, self.tokenizer, self.config['model']['data']['max_length'])
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
            prompts, responses = self._prepare_data(data_path)
            train_dataset, val_dataset = self._create_datasets(prompts, responses)
            
            # Setup training arguments
            learning_rate = self.config['model']['training']['learning_rate']
            logger.info(f"Learning rate: {learning_rate} (type: {type(learning_rate)})")
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config['model']['training']['epochs'],
                per_device_train_batch_size=self.config['model']['training']['batch_size'],
                per_device_eval_batch_size=self.config['model']['inference']['batch_size'],
                learning_rate=float(learning_rate),
                weight_decay=self.config['model']['training']['weight_decay'],
                logging_dir=self.config['model']['logging']['tensorboard_dir'],
                logging_steps=self.config['model']['training']['logging_steps'],
                save_steps=self.config['model']['training']['save_steps'],
                eval_steps=self.config['model']['training']['eval_steps'],
                save_total_limit=2,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False,
                report_to=["wandb"],
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
            if self.tokenizer is not None:
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
    
    def save_to_s3(self, local_path: str, s3_bucket: Optional[str] = None, s3_prefix: Optional[str] = None):
        """Save model to S3"""
        try:
            if not s3_bucket:
                s3_bucket = self.config['model']['aws']['s3_bucket']
            if not s3_prefix:
                s3_prefix = self.config['model']['aws']['s3_prefix']
            
            if not s3_bucket or not s3_prefix:
                raise ValueError("S3 bucket and prefix must be provided either as parameters or in config")
            
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
