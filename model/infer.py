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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from loguru import logger
import boto3
from botocore.exceptions import ClientError


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
        """Load the trained model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config['model']['base_model'],
                num_labels=self.config['model']['classification']['num_labels'],
                problem_type="multi_label_classification"
            )
            
            # Load PEFT model
            self.peft_model = PeftModel.from_pretrained(self.model, self.model_path, local_files_only=True)
            self.peft_model.eval()
            
            # Move to device
            self.peft_model.to(self.device)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
    
    def predict(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a business"""
        try:
            # Preprocess input
            inputs = self.preprocess_input(business_data)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.peft_model(**inputs)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits)
            
            # Convert to numpy
            probs = probabilities.cpu().numpy()[0]
            
            # Extract predictions
            risk_probs = probs[:3]  # First 3 are risk levels
            funding_probs = probs[3:]  # Last 3 are funding levels
            
            # Get predictions
            risk_prediction = np.argmax(risk_probs)
            funding_prediction = np.argmax(funding_probs)
            
            # Get confidence scores
            risk_confidence = float(risk_probs[risk_prediction])
            funding_confidence = float(funding_probs[funding_prediction])
            
            # Map to labels
            risk_labels = self.config['model']['classification']['risk_labels']
            funding_labels = self.config['model']['classification']['funding_labels']
            
            risk_label = risk_labels[risk_prediction]
            funding_label = funding_labels[funding_prediction]
            
            # Calculate overall confidence
            overall_confidence = (risk_confidence + funding_confidence) / 2
            
            # Create result
            result = {
                'business_name': business_data.get('business_name', 'Unknown'),
                'risk_assessment': {
                    'level': risk_label,
                    'confidence': risk_confidence,
                    'probabilities': {
                        'low': float(risk_probs[0]),
                        'medium': float(risk_probs[1]),
                        'high': float(risk_probs[2])
                    }
                },
                'funding_recommendation': {
                    'decision': funding_label,
                    'confidence': funding_confidence,
                    'probabilities': {
                        'yes': float(funding_probs[0]),
                        'no': float(funding_probs[1]),
                        'maybe': float(funding_probs[2])
                    }
                },
                'overall_confidence': overall_confidence,
                'model_version': self.config['model']['version']
            }
            
            logger.info(f"Prediction completed for {result['business_name']}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
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
        """Analyze a business and provide comprehensive recommendations"""
        try:
            # Make prediction
            prediction = self.inference.predict(business_data)
            
            # Add analysis insights
            analysis = self._generate_insights(business_data, prediction)
            
            # Combine results
            result = {
                **prediction,
                'analysis': analysis,
                'timestamp': str(pd.Timestamp.now()),
                'recommendations': self._generate_recommendations(prediction, business_data)
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