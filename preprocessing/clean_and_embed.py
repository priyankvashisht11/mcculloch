"""
Preprocessing Module for LCF Group Funding Recommendation System

Handles:
- Text cleaning and normalization
- Named Entity Recognition (NER)
- Sentence embedding generation
- Data validation and transformation
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pydantic import BaseModel, Field
import pickle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessedBusiness(BaseModel):
    """Pydantic model for preprocessed business data"""
    business_name: str
    cleaned_description: str
    embedding: List[float]
    entities: Dict[str, List[str]]
    features: Dict[str, Any]
    risk_score: Optional[float] = None
    funding_recommendation: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass


class TextCleaner:
    """Text cleaning and normalization utilities"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (optional - can be kept for financial data)
        # text = re.sub(r'\d+', '', text)
        
        # Tokenize and remove stop words
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if not text:
            return []
        
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        if not text:
            return ""
        
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)


class EntityExtractor:
    """Named Entity Recognition and extraction"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"SpaCy model {model_name} not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        if not text:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text.strip()
            
            if entity_type not in entities:
                entities[entity_type] = []
            
            if entity_text not in entities[entity_type]:
                entities[entity_type].append(entity_text)
        
        return entities
    
    def extract_business_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract business-specific entities"""
        entities = self.extract_entities(text)
        
        # Focus on business-relevant entities
        business_entities = {
            'ORG': entities.get('ORG', []),  # Organizations
            'GPE': entities.get('GPE', []),  # Geographical locations
            'MONEY': entities.get('MONEY', []),  # Monetary values
            'DATE': entities.get('DATE', []),  # Dates
            'CARDINAL': entities.get('CARDINAL', []),  # Numbers
            'PRODUCT': entities.get('PRODUCT', []),  # Products
        }
        
        return business_entities


class EmbeddingGenerator:
    """Generate sentence embeddings for semantic search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text:
            return []
        
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [[] for _ in texts]


class FeatureExtractor:
    """Extract and normalize features from business data"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
    
    def extract_numerical_features(self, business_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize numerical features"""
        features = {}
        
        # Revenue features
        revenue = business_data.get('revenue', 0)
        features['revenue'] = float(revenue) if revenue else 0.0
        features['revenue_log'] = np.log1p(features['revenue']) if features['revenue'] > 0 else 0.0
        
        # Employee count
        employee_count = business_data.get('employee_count', 0)
        features['employee_count'] = float(employee_count) if employee_count else 0.0
        features['employee_count_log'] = np.log1p(features['employee_count']) if features['employee_count'] > 0 else 0.0
        
        # Years active
        years_active = business_data.get('years_active', 0)
        features['years_active'] = float(years_active) if years_active else 0.0
        
        # Past funding
        past_funding = business_data.get('past_funding', 0)
        features['past_funding'] = float(past_funding) if past_funding else 0.0
        features['past_funding_log'] = np.log1p(features['past_funding']) if features['past_funding'] > 0 else 0.0
        
        # Derived features
        features['revenue_per_employee'] = (
            features['revenue'] / features['employee_count'] 
            if features['employee_count'] > 0 else 0.0
        )
        
        features['funding_to_revenue_ratio'] = (
            features['past_funding'] / features['revenue'] 
            if features['revenue'] > 0 else 0.0
        )
        
        return features
    
    def extract_categorical_features(self, business_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract and encode categorical features"""
        features = {}
        
        # Domain/Industry encoding
        domain = business_data.get('domain', 'unknown')
        if 'domain' not in self.label_encoders:
            self.label_encoders['domain'] = LabelEncoder()
            # Fit with common domains (in production, fit on training data)
            common_domains = ['technology', 'healthcare', 'finance', 'retail', 'manufacturing', 'unknown']
            self.label_encoders['domain'].fit(common_domains)
        
        try:
            features['domain_encoded'] = self.label_encoders['domain'].transform([domain])[0]
        except ValueError:
            features['domain_encoded'] = self.label_encoders['domain'].transform(['unknown'])[0]
        
        # Location encoding
        location = business_data.get('location', 'unknown')
        if 'location' not in self.label_encoders:
            self.label_encoders['location'] = LabelEncoder()
            # Fit with common locations
            common_locations = ['california', 'new york', 'texas', 'florida', 'unknown']
            self.label_encoders['location'].fit(common_locations)
        
        try:
            features['location_encoded'] = self.label_encoders['location'].transform([location.lower()])[0]
        except ValueError:
            features['location_encoded'] = self.label_encoders['location'].transform(['unknown'])[0]
        
        return features
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract text-based features"""
        if not text:
            return {}
        
        features = {}
        
        # Text length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0.0
        
        # Vocabulary richness
        unique_words = set(words)
        features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0.0
        
        return features
    
    def normalize_features(self, features: Dict[str, float], feature_type: str = "numerical") -> Dict[str, float]:
        """Normalize features using StandardScaler"""
        if feature_type not in self.scalers:
            self.scalers[feature_type] = StandardScaler()
            # In production, fit on training data
            # For now, we'll use a simple min-max normalization
        
        # Simple min-max normalization for demonstration
        normalized = {}
        for key, value in features.items():
            # Apply log transformation for skewed features
            if 'revenue' in key or 'funding' in key:
                normalized[key] = np.log1p(value) if value > 0 else 0.0
            else:
                normalized[key] = value
        
        return normalized


class BusinessPreprocessor:
    """Main preprocessing class that orchestrates all preprocessing steps"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.text_cleaner = TextCleaner()
        self.entity_extractor = EntityExtractor()
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.feature_extractor = FeatureExtractor()
        
    def preprocess_business(self, business_data: Dict[str, Any]) -> PreprocessedBusiness:
        """Preprocess a single business profile"""
        try:
            # Extract text data
            description = business_data.get('description', '')
            business_name = business_data.get('business_name', 'Unknown Business')
            
            # Clean text
            cleaned_description = self.text_cleaner.clean_text(description)
            
            # Extract entities
            entities = self.entity_extractor.extract_business_entities(description)
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(cleaned_description)
            
            # Extract features
            numerical_features = self.feature_extractor.extract_numerical_features(business_data)
            categorical_features = self.feature_extractor.extract_categorical_features(business_data)
            text_features = self.feature_extractor.extract_text_features(cleaned_description)
            
            # Combine all features
            all_features = {
                **numerical_features,
                **categorical_features,
                **text_features
            }
            
            # Normalize features
            normalized_features = self.feature_extractor.normalize_features(all_features)
            
            return PreprocessedBusiness(
                business_name=business_name,
                cleaned_description=cleaned_description,
                embedding=embedding,
                entities=entities,
                features=normalized_features
            )
            
        except Exception as e:
            logger.error(f"Error preprocessing business {business_data.get('business_name', 'Unknown')}: {e}")
            raise PreprocessingError(f"Failed to preprocess business: {e}")
    
    def preprocess_batch(self, business_data_list: List[Dict[str, Any]]) -> List[PreprocessedBusiness]:
        """Preprocess multiple business profiles"""
        preprocessed = []
        
        for business_data in business_data_list:
            try:
                preprocessed_business = self.preprocess_business(business_data)
                preprocessed.append(preprocessed_business)
            except Exception as e:
                logger.error(f"Error preprocessing business in batch: {e}")
                continue
        
        logger.info(f"Successfully preprocessed {len(preprocessed)} out of {len(business_data_list)} businesses")
        return preprocessed
    
    def save_preprocessed_data(self, preprocessed_data: List[PreprocessedBusiness], 
                             output_path: str) -> str:
        """Save preprocessed data to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            serializable_data = []
            for item in preprocessed_data:
                serializable_item = item.dict()
                serializable_item['embedding'] = item.embedding
                serializable_data.append(serializable_item)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            logger.info(f"Saved preprocessed data to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
            raise PreprocessingError(f"Failed to save preprocessed data: {e}")
    
    def load_preprocessed_data(self, file_path: str) -> List[PreprocessedBusiness]:
        """Load preprocessed data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            preprocessed_data = []
            for item in data:
                preprocessed_business = PreprocessedBusiness(**item)
                preprocessed_data.append(preprocessed_business)
            
            logger.info(f"Loaded {len(preprocessed_data)} preprocessed businesses from {file_path}")
            return preprocessed_data
            
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            raise PreprocessingError(f"Failed to load preprocessed data: {e}")


def main():
    """Example usage of the preprocessing module"""
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
    
    # Initialize preprocessor
    preprocessor = BusinessPreprocessor()
    
    try:
        # Preprocess single business
        preprocessed = preprocessor.preprocess_business(sample_business)
        
        print(f"Preprocessed business: {preprocessed.business_name}")
        print(f"Cleaned description: {preprocessed.cleaned_description[:100]}...")
        print(f"Embedding length: {len(preprocessed.embedding)}")
        print(f"Entities: {preprocessed.entities}")
        print(f"Features: {list(preprocessed.features.keys())}")
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data([preprocessed], "data/preprocessed/sample_business.json")
        
    except Exception as e:
        logger.error(f"Error in preprocessing example: {e}")


if __name__ == "__main__":
    main() 