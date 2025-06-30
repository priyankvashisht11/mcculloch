"""
Inference Module for LCF Group Funding Recommendation System

Handles model loading and prediction for business funding recommendations.
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig
from loguru import logger
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from transformers.utils.quantization_config import BitsAndBytesConfig


class FundingInference:
    """Inference class for funding recommendations"""
    
    def __init__(self, model_path: str = "models/lora_finetuned", config_path: str = "model/config/model_config.yaml"):
        # Ensure model_path is absolute within the container
        if not model_path.startswith('/'):
            self.model_path = f"/app/{model_path}"
        else:
            self.model_path = model_path
            
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def load_model(self):
        """Load the base model and LoRA adapter for LLM inference"""
        try:
            print("🔍 Starting model loading process...")
            # Load tokenizer from the LoRA adapter directory (for any special tokens)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Tokenizer loaded successfully")
            # Load base model from models/mistral-7b-instruct-v0.2
            base_model_path = "/app/models/mistral-7b-instruct-v0.2"
            print(f"🤖 Loading base model from: {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)
            # Load LoRA adapter and merge
            print(f"🔧 Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path, local_files_only=True)
            self.model.eval()
            print(f"✅ Model with LoRA adapter loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_model_from_s3(self, s3_bucket: str, s3_prefix: str, local_path: str = "models/lora_finetuned"):
        """Load model from S3"""
        try:
            s3_client = boto3.client('s3')
            
            # Create local directory
            os.makedirs(local_path, exist_ok=True)
            
            # List objects in S3
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
            
            # Download files
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                        
                        # Create directory if needed
                        os.makedirs(os.path.dirname(local_file), exist_ok=True)
                        
                        # Download file
                        s3_client.download_file(s3_bucket, s3_key, local_file)
                        logger.info(f"Downloaded {s3_key} to {local_file}")
            
            # Load model from local path
            self.model_path = local_path
            self.load_model()
            
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading from S3: {e}")
            raise
    
    def preprocess_input(self, business_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess input data for inference"""
        try:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer is not loaded. Call load_model() before preprocessing input.")
            # Extract text description
            description = business_data.get('description', '')
            if not description:
                description = f"{business_data.get('business_name', '')} {business_data.get('domain', '')}"
            
            # Tokenize text
            encoding = self.tokenizer(
                description,
                truncation=True,
                padding='max_length',
                max_length=self.config['model']['data']['max_length'],
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise
    
    def build_prompt(self, business_data: Dict[str, Any]) -> str:
        """Builds a prompt for LLM generation from business data."""
        # Use the same format as your fine-tuning data
        instruction = business_data.get('instruction', 'Evaluate the business profile')
        # Build input string from business_data fields
        input_lines = []
        for k, v in business_data.items():
            if k != 'instruction':
                input_lines.append(f"{k}: {v}")
        input_str = "\n".join(input_lines)
        prompt = f"<s>[INST] {instruction}\n{input_str} [/INST]"
        return prompt

    def predict(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM output for a business profile."""
        try:
            prompt = self.build_prompt(business_data)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Remove the prompt from the output to get only the generated answer
            answer = generated.replace(prompt, "").strip()
            return {
                'llm_output': answer,
                'model_version': self.config.get('model', {}).get('version', 'unknown')
            }
        except Exception as e:
            logger.error(f"Error generating LLM output: {e}")
            raise
    
    def predict_batch(self, business_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple businesses"""
        results = []
        
        for business_data in business_data_list:
            try:
                result = self.predict(business_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for {business_data.get('business_name', 'Unknown')}: {e}")
                # Add error result
                results.append({
                    'business_name': business_data.get('business_name', 'Unknown'),
                    'error': str(e),
                    'risk_assessment': {'level': 'unknown', 'confidence': 0.0},
                    'funding_recommendation': {'decision': 'unknown', 'confidence': 0.0},
                    'overall_confidence': 0.0
                })
        
        logger.info(f"Batch prediction completed: {len(results)} results")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            info = {
                'model_path': self.model_path,
                'base_model': self.config['model']['base_model'],
                'device': str(self.device),
                'num_labels': self.config['model']['classification']['num_labels'],
                'max_length': self.config['model']['data']['max_length'],
                'version': self.config['model']['version'],
                'lora_config': self.config['model']['lora_config'],
                'risk_labels': self.config['model']['classification']['risk_labels'],
                'funding_labels': self.config['model']['classification']['funding_labels']
            }
            
            # Add model parameters count if available
            if self.peft_model:
                total_params = sum(p.numel() for p in self.peft_model.parameters())
                trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
                info['total_parameters'] = total_params
                info['trainable_parameters'] = trainable_params
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise
    
    def save_prediction(self, prediction: Dict[str, Any], output_path: str):
        """Save prediction to file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(prediction, f, indent=2, default=str)
            
            logger.info(f"Prediction saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            raise


class FundingAnalyzer:
    """High-level analyzer for funding recommendations"""
    
    def __init__(self, model_path: str = "models/lora_finetuned"):
        # Ensure model_path is absolute within the container
        if not model_path.startswith('/'):
            absolute_model_path = f"/app/{model_path}"
        else:
            absolute_model_path = model_path
            
        self.inference = FundingInference(model_path=absolute_model_path)
        try:
            self.inference.load_model()
        except Exception as e:
            logger.error(f"Failed to load model during initialization: {e}")
            raise
    
    def analyze_business(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a business and provide LLM-based generative analysis"""
        try:
            # Generate LLM output
            prediction = self.inference.predict(business_data)
            result = {
                'llm_output': prediction['llm_output'],
                'timestamp': str(pd.Timestamp.now()),
                'model_version': prediction.get('model_version', 'unknown')
            }
            return result
        except Exception as e:
            logger.error(f"Error analyzing business: {e}")
            raise
    
    def _generate_insights(self, business_data: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights based on business data and prediction"""
        insights = {
            'key_factors': [],
            'strengths': [],
            'concerns': [],
            'market_position': 'unknown'
        }
        
        # Analyze revenue
        revenue = business_data.get('revenue', 0)
        if revenue > 1000000:
            insights['strengths'].append("Strong revenue generation")
            insights['market_position'] = "established"
        elif revenue > 500000:
            insights['strengths'].append("Moderate revenue growth")
            insights['market_position'] = "growing"
        else:
            insights['concerns'].append("Low revenue may indicate market challenges")
            insights['market_position'] = "early_stage"
        
        # Analyze employee count
        employee_count = business_data.get('employee_count', 0)
        if employee_count > 50:
            insights['strengths'].append("Established team structure")
        elif employee_count > 10:
            insights['key_factors'].append("Growing team size")
        else:
            insights['concerns'].append("Small team may limit scalability")
        
        # Analyze years active
        years_active = business_data.get('years_active', 0)
        if years_active > 5:
            insights['strengths'].append("Proven track record")
        elif years_active > 2:
            insights['key_factors'].append("Established business model")
        else:
            insights['concerns'].append("Limited operating history")
        
        # Analyze domain
        domain = business_data.get('domain', '').lower()
        if domain in ['technology', 'software', 'ai']:
            insights['key_factors'].append("High-growth technology sector")
        elif domain in ['healthcare', 'finance']:
            insights['key_factors'].append("Regulated industry with barriers to entry")
        
        return insights
    
    def _generate_recommendations(self, prediction: Dict[str, Any], business_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        risk_level = prediction['risk_assessment']['level']
        funding_decision = prediction['funding_recommendation']['decision']
        
        if risk_level == 'high':
            recommendations.append("Consider additional risk mitigation strategies")
            recommendations.append("Implement stronger financial controls")
        
        if funding_decision == 'yes':
            recommendations.append("Proceed with funding application")
            recommendations.append("Prepare detailed business plan")
        elif funding_decision == 'maybe':
            recommendations.append("Address identified risk factors before proceeding")
            recommendations.append("Consider smaller initial funding amount")
        else:
            recommendations.append("Focus on business model validation")
            recommendations.append("Consider alternative funding sources")
        
        # Add domain-specific recommendations
        domain = business_data.get('domain', '').lower()
        if domain == 'technology':
            recommendations.append("Highlight technological innovation and IP")
        elif domain == 'healthcare':
            recommendations.append("Ensure regulatory compliance documentation")
        
        return recommendations


def main():
    """Example usage of the inference module"""
    # Sample business data
    sample_business = {
        "business_name": "TechStart Inc",
        "domain": "Technology",
        "years_active": 3,
        "revenue": 500000,
        "employee_count": 15,
        "location": "California",
        "description": "TechStart Inc is a leading AI-powered SaaS platform that helps businesses automate their workflow. Founded in 2020, we have grown to serve over 500 clients across the United States."
    }
    
    try:
        # Initialize analyzer
        analyzer = FundingAnalyzer()
        
        # Analyze business
        result = analyzer.analyze_business(sample_business)
        
        print("=== Funding Analysis Results ===")
        print(f"Business: {result['business_name']}")
        print(f"Risk Level: {result['risk_assessment']['level']} (Confidence: {result['risk_assessment']['confidence']:.2f})")
        print(f"Funding Decision: {result['funding_recommendation']['decision']} (Confidence: {result['funding_recommendation']['confidence']:.2f})")
        print(f"Overall Confidence: {result['overall_confidence']:.2f}")
        
        print("\n=== Key Insights ===")
        for category, items in result['analysis'].items():
            if isinstance(items, list) and items:
                print(f"{category.title()}: {', '.join(items)}")
        
        print("\n=== Recommendations ===")
        for rec in result['recommendations']:
            print(f"• {rec}")
        
        # Save result
        analyzer.inference.save_prediction(result, "predictions/techstart_analysis.json")
        
    except Exception as e:
        logger.error(f"Error in inference example: {e}")


if __name__ == "__main__":
    main() 