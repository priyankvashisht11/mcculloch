"""
Data Ingestion Module for LCF Group Funding Recommendation System

Handles various input formats:
- CSV/Excel files
- PDF documents
- Text files
- API data
- Structured JSON data
"""

import os
import json
import pandas as pd
import PyPDF2
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import requests
from pydantic import BaseModel, Field
import openpyxl
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BusinessProfile(BaseModel):
    """Pydantic model for business profile data"""
    business_name: str = Field(..., description="Name of the business")
    domain: Optional[str] = Field(None, description="Business domain/sector")
    years_active: Optional[float] = Field(None, description="Years in business")
    revenue: Optional[float] = Field(None, description="Annual revenue")
    past_funding: Optional[float] = Field(None, description="Previous funding received")
    sentiment: Optional[str] = Field(None, description="Business sentiment")
    location: Optional[str] = Field(None, description="Business location")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    description: Optional[str] = Field(None, description="Business description")
    website: Optional[str] = Field(None, description="Business website")
    industry: Optional[str] = Field(None, description="Industry classification")
    risk_factors: Optional[List[str]] = Field(None, description="Identified risk factors")
    growth_potential: Optional[str] = Field(None, description="Growth potential assessment")
    
    class Config:
        extra = "allow"  # Allow additional fields


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass


class DataIngestion:
    """Main data ingestion class"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def ingest_csv(self, file_path: str) -> List[BusinessProfile]:
        """Ingest data from CSV file"""
        try:
            logger.info(f"Ingesting CSV file: {file_path}")
            df = pd.read_csv(file_path)
            
            profiles = []
            for _, row in df.iterrows():
                profile_data = row.to_dict()
                # Clean and validate data
                profile_data = self._clean_data(profile_data)
                profile = BusinessProfile(**profile_data)
                profiles.append(profile)
                
            logger.info(f"Successfully ingested {len(profiles)} profiles from CSV")
            return profiles
            
        except Exception as e:
            logger.error(f"Error ingesting CSV file: {e}")
            raise DataIngestionError(f"Failed to ingest CSV: {e}")
    
    def ingest_excel(self, file_path: str, sheet_name: Optional[str] = None) -> List[BusinessProfile]:
        """Ingest data from Excel file"""
        try:
            logger.info(f"Ingesting Excel file: {file_path}")
            
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            profiles = []
            for _, row in df.iterrows():
                profile_data = row.to_dict()
                profile_data = self._clean_data(profile_data)
                profile = BusinessProfile(**profile_data)
                profiles.append(profile)
                
            logger.info(f"Successfully ingested {len(profiles)} profiles from Excel")
            return profiles
            
        except Exception as e:
            logger.error(f"Error ingesting Excel file: {e}")
            raise DataIngestionError(f"Failed to ingest Excel: {e}")
    
    def ingest_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF and create business profile"""
        try:
            logger.info(f"Extracting text from PDF: {file_path}")
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # Extract business information using NLP/pattern matching
            profile_data = self._extract_business_info_from_text(text)
            profile = BusinessProfile(**profile_data)
            
            logger.info("Successfully extracted business profile from PDF")
            return {"profile": profile, "raw_text": text}
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise DataIngestionError(f"Failed to process PDF: {e}")
    
    def ingest_text(self, file_path: str) -> Dict[str, Any]:
        """Extract business information from text file"""
        try:
            logger.info(f"Processing text file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            profile_data = self._extract_business_info_from_text(text)
            profile = BusinessProfile(**profile_data)
            
            logger.info("Successfully extracted business profile from text")
            return {"profile": profile, "raw_text": text}
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise DataIngestionError(f"Failed to process text file: {e}")
    
    def ingest_api_data(self, api_url: str, headers: Optional[Dict] = None) -> List[BusinessProfile]:
        """Ingest data from API endpoint"""
        try:
            logger.info(f"Fetching data from API: {api_url}")
            
            response = requests.get(api_url, headers=headers or {})
            response.raise_for_status()
            
            data = response.json()
            
            profiles = []
            if isinstance(data, list):
                for item in data:
                    item = self._clean_data(item)
                    profile = BusinessProfile(**item)
                    profiles.append(profile)
            elif isinstance(data, dict):
                data = self._clean_data(data)
                profile = BusinessProfile(**data)
                profiles.append(profile)
            
            logger.info(f"Successfully ingested {len(profiles)} profiles from API")
            return profiles
            
        except Exception as e:
            logger.error(f"Error fetching data from API: {e}")
            raise DataIngestionError(f"Failed to fetch API data: {e}")
    
    def ingest_json(self, file_path: str) -> List[BusinessProfile]:
        """Ingest data from JSON file"""
        try:
            logger.info(f"Ingesting JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            profiles = []
            if isinstance(data, list):
                for item in data:
                    item = self._clean_data(item)
                    profile = BusinessProfile(**item)
                    profiles.append(profile)
            elif isinstance(data, dict):
                data = self._clean_data(data)
                profile = BusinessProfile(**data)
                profiles.append(profile)
            
            logger.info(f"Successfully ingested {len(profiles)} profiles from JSON")
            return profiles
            
        except Exception as e:
            logger.error(f"Error ingesting JSON file: {e}")
            raise DataIngestionError(f"Failed to ingest JSON: {e}")
    
    def _clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize data"""
        cleaned = {}
        
        for key, value in data.items():
            if pd.isna(value) or value == "" or value == "nan":
                continue
                
            # Convert column names to snake_case
            clean_key = key.lower().replace(" ", "_").replace("-", "_")
            
            # Type conversion and cleaning
            if isinstance(value, str):
                value = value.strip()
                # Convert numeric strings
                if value.replace(".", "").replace("-", "").isdigit():
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
            
            cleaned[clean_key] = value
        
        return cleaned
    
    def _extract_business_info_from_text(self, text: str) -> Dict[str, Any]:
        """Extract business information from unstructured text using NLP patterns"""
        # This is a simplified extraction - in production, use more sophisticated NLP
        profile_data = {
            "business_name": "",
            "description": text[:500] if len(text) > 500 else text
        }
        
        # Simple pattern matching for common business information
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract business name (usually in first few lines)
            if not profile_data["business_name"] and len(line) > 3 and len(line) < 100:
                if any(word in line.lower() for word in ['inc', 'llc', 'corp', 'company', 'ltd']):
                    profile_data["business_name"] = line
            
            # Extract revenue information
            if any(word in line.lower() for word in ['revenue', 'sales', '$']):
                # Extract numbers
                import re
                numbers = re.findall(r'\$?[\d,]+(?:\.\d{2})?', line)
                if numbers:
                    profile_data["revenue"] = str(float(numbers[0].replace('$', '').replace(',', '')))
            
            # Extract employee count
            if any(word in line.lower() for word in ['employees', 'staff', 'team']):
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    profile_data["employee_count"] = str(int(numbers[0]))
        
        return profile_data
    
    def save_profiles(self, profiles: List[BusinessProfile], filename: str) -> str:
        """Save processed profiles to JSON file"""
        try:
            output_path = self.output_dir / filename
            
            # Convert to dict for JSON serialization
            data = [profile.dict() for profile in profiles]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(profiles)} profiles to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")
            raise DataIngestionError(f"Failed to save profiles: {e}")
    
    def batch_ingest(self, input_dir: str, file_pattern: str = "*") -> List[BusinessProfile]:
        """Batch ingest multiple files from a directory"""
        input_path = Path(input_dir)
        all_profiles = []
        
        for file_path in input_path.glob(file_pattern):
            try:
                if file_path.suffix.lower() == '.csv':
                    profiles = self.ingest_csv(str(file_path))
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    profiles = self.ingest_excel(str(file_path))
                elif file_path.suffix.lower() == '.pdf':
                    result = self.ingest_pdf(str(file_path))
                    profiles = [result["profile"]]
                elif file_path.suffix.lower() == '.json':
                    profiles = self.ingest_json(str(file_path))
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    result = self.ingest_text(str(file_path))
                    profiles = [result["profile"]]
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                all_profiles.extend(profiles)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Batch ingestion completed. Total profiles: {len(all_profiles)}")
        return all_profiles


def main():
    """Example usage of the data ingestion module"""
    # Initialize data ingestion
    ingestion = DataIngestion()
    
    # Example: Ingest from different sources
    try:
        # CSV ingestion
        if os.path.exists("data/sample_businesses.csv"):
            profiles = ingestion.ingest_csv("data/sample_businesses.csv")
            ingestion.save_profiles(profiles, "processed_businesses.json")
        
        # PDF ingestion
        if os.path.exists("data/business_profile.pdf"):
            result = ingestion.ingest_pdf("data/business_profile.pdf")
            ingestion.save_profiles([result["profile"]], "pdf_extracted.json")
        
        # Batch ingestion
        if os.path.exists("data/raw"):
            all_profiles = ingestion.batch_ingest("data/raw", "*.csv")
            ingestion.save_profiles(all_profiles, "batch_processed.json")
            
    except Exception as e:
        logger.error(f"Error in main ingestion: {e}")


if __name__ == "__main__":
    main() 