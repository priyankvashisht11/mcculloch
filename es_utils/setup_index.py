"""
Elasticsearch Setup Module for LCF Group Funding Recommendation System

Handles:
- Index creation with dense vector mapping
- Business data indexing
- Search configuration
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError, NotFoundError
from loguru import logger
import numpy as np


class ElasticsearchSetup:
    """Elasticsearch setup and management class"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 9200, 
                 username: str = "elastic", 
                 password: str = "changeme",
                 use_ssl: bool = False):
        
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        
        # Initialize Elasticsearch client
        if username and password:
            self.es = Elasticsearch(
                [f"http://{host}:{port}"],
                basic_auth=(username, password),
                verify_certs=False,
                ssl_show_warn=False
            )
        else:
            self.es = Elasticsearch([f"http://{host}:{port}"])
        
        # Index configuration
        self.index_name = "lcf_businesses"
        self.index_mapping = self._get_index_mapping()
        
        logger.info(f"Initialized Elasticsearch client for {host}:{port}")
    
    def _get_index_mapping(self) -> Dict[str, Any]:
        """Get the index mapping with dense vector support"""
        return {
            "mappings": {
                "properties": {
                    "business_name": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "domain": {
                        "type": "keyword"
                    },
                    "industry": {
                        "type": "keyword"
                    },
                    "location": {
                        "type": "keyword"
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "cleaned_description": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,  # all-MiniLM-L6-v2 embedding dimension
                        "index": True,
                        "similarity": "cosine"
                    },
                    "features": {
                        "properties": {
                            "revenue": {"type": "float"},
                            "revenue_log": {"type": "float"},
                            "employee_count": {"type": "integer"},
                            "employee_count_log": {"type": "float"},
                            "years_active": {"type": "float"},
                            "past_funding": {"type": "float"},
                            "past_funding_log": {"type": "float"},
                            "revenue_per_employee": {"type": "float"},
                            "funding_to_revenue_ratio": {"type": "float"},
                            "domain_encoded": {"type": "integer"},
                            "location_encoded": {"type": "integer"},
                            "text_length": {"type": "integer"},
                            "word_count": {"type": "integer"},
                            "sentence_count": {"type": "integer"},
                            "avg_word_length": {"type": "float"},
                            "vocabulary_richness": {"type": "float"}
                        }
                    },
                    "entities": {
                        "properties": {
                            "ORG": {"type": "keyword"},
                            "GPE": {"type": "keyword"},
                            "MONEY": {"type": "keyword"},
                            "DATE": {"type": "keyword"},
                            "CARDINAL": {"type": "keyword"},
                            "PRODUCT": {"type": "keyword"}
                        }
                    },
                    "risk_assessment": {
                        "properties": {
                            "level": {"type": "keyword"},
                            "confidence": {"type": "float"},
                            "probabilities": {
                                "properties": {
                                    "low": {"type": "float"},
                                    "medium": {"type": "float"},
                                    "high": {"type": "float"}
                                }
                            }
                        }
                    },
                    "funding_recommendation": {
                        "properties": {
                            "decision": {"type": "keyword"},
                            "confidence": {"type": "float"},
                            "probabilities": {
                                "properties": {
                                    "yes": {"type": "float"},
                                    "no": {"type": "float"},
                                    "maybe": {"type": "float"}
                                }
                            }
                        }
                    },
                    "overall_confidence": {"type": "float"},
                    "timestamp": {"type": "date"},
                    "metadata": {
                        "properties": {
                            "source": {"type": "keyword"},
                            "version": {"type": "keyword"},
                            "tags": {"type": "keyword"}
                        }
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            }
        }
    
    def check_connection(self) -> bool:
        """Check Elasticsearch connection"""
        try:
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False
    
    def create_index(self, force_recreate: bool = False) -> bool:
        """Create the business index"""
        try:
            # Check if index exists
            if self.es.indices.exists(index=self.index_name):
                if force_recreate:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.es.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index {self.index_name} already exists")
                    return True
            
            # Create index
            logger.info(f"Creating index: {self.index_name}")
            response = self.es.indices.create(
                index=self.index_name,
                body=self.index_mapping
            )
            
            if response['acknowledged']:
                logger.info(f"Successfully created index: {self.index_name}")
                return True
            else:
                logger.error("Failed to create index")
                return False
                
        except RequestError as e:
            logger.error(f"Error creating index: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating index: {e}")
            return False
    
    def index_business(self, business_data: Dict[str, Any]) -> bool:
        """Index a single business document"""
        try:
            # Prepare document
            doc = self._prepare_document(business_data)
            
            # Index document
            response = self.es.index(
                index=self.index_name,
                body=doc
            )
            
            if response['result'] in ['created', 'updated']:
                logger.info(f"Indexed business: {business_data.get('business_name', 'Unknown')}")
                return True
            else:
                logger.error(f"Failed to index business: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error indexing business: {e}")
            return False
    
    def index_businesses_batch(self, business_data_list: List[Dict[str, Any]]) -> int:
        """Index multiple business documents in batch"""
        try:
            # Prepare bulk operations
            bulk_data = []
            for business_data in business_data_list:
                # Prepare document
                doc = self._prepare_document(business_data)
                
                # Add index operation
                bulk_data.append({"index": {"_index": self.index_name}})
                bulk_data.append(doc)
            
            if not bulk_data:
                logger.warning("No data to index")
                return 0
            
            # Execute bulk operation
            response = self.es.bulk(body=bulk_data)
            
            # Check results
            success_count = 0
            error_count = 0
            
            for item in response['items']:
                if 'index' in item and item['index']['status'] in [200, 201]:
                    success_count += 1
                else:
                    error_count += 1
                    logger.error(f"Bulk indexing error: {item}")
            
            logger.info(f"Bulk indexing completed: {success_count} successful, {error_count} failed")
            return success_count
            
        except Exception as e:
            logger.error(f"Error in bulk indexing: {e}")
            return 0
    
    def _prepare_document(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare business data for indexing"""
        doc = {
            "business_name": business_data.get("business_name", ""),
            "domain": business_data.get("domain", ""),
            "industry": business_data.get("industry", ""),
            "location": business_data.get("location", ""),
            "description": business_data.get("description", ""),
            "cleaned_description": business_data.get("cleaned_description", ""),
            "features": business_data.get("features", {}),
            "entities": business_data.get("entities", {}),
            "timestamp": business_data.get("timestamp", ""),
            "metadata": {
                "source": business_data.get("source", "manual"),
                "version": business_data.get("version", "1.0"),
                "tags": business_data.get("tags", [])
            }
        }
        
        # Add embedding if available
        if "embedding" in business_data:
            doc["embedding"] = business_data["embedding"]
        
        # Add prediction results if available
        if "risk_assessment" in business_data:
            doc["risk_assessment"] = business_data["risk_assessment"]
        
        if "funding_recommendation" in business_data:
            doc["funding_recommendation"] = business_data["funding_recommendation"]
        
        if "overall_confidence" in business_data:
            doc["overall_confidence"] = business_data["overall_confidence"]
        
        return doc
    
    def load_data_from_file(self, file_path: str) -> int:
        """Load business data from JSON file and index it"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return self.index_businesses_batch(data)
            elif isinstance(data, dict):
                success = self.index_business(data)
                return 1 if success else 0
            else:
                logger.error("Invalid data format in file")
                return 0
                
        except Exception as e:
            logger.error(f"Error loading data from file: {e}")
            return 0
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            return {
                "document_count": stats['indices'][self.index_name]['total']['docs']['count'],
                "index_size": stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                "indexing_rate": stats['indices'][self.index_name]['total']['indexing']['index_total']
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    def delete_index(self) -> bool:
        """Delete the business index"""
        try:
            if self.es.indices.exists(index=self.index_name):
                response = self.es.indices.delete(index=self.index_name)
                if response['acknowledged']:
                    logger.info(f"Successfully deleted index: {self.index_name}")
                    return True
                else:
                    logger.error("Failed to delete index")
                    return False
            else:
                logger.info(f"Index {self.index_name} does not exist")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    def update_index_settings(self, settings: Dict[str, Any]) -> bool:
        """Update index settings"""
        try:
            response = self.es.indices.put_settings(
                index=self.index_name,
                body=settings
            )
            
            if response['acknowledged']:
                logger.info("Successfully updated index settings")
                return True
            else:
                logger.error("Failed to update index settings")
                return False
                
        except Exception as e:
            logger.error(f"Error updating index settings: {e}")
            return False


def main():
    """Example usage of the Elasticsearch setup module"""
    # Initialize setup
    es_setup = ElasticsearchSetup(
        host="localhost",
        port=9200,
        username="elastic",
        password="changeme"
    )
    
    try:
        # Check connection
        if not es_setup.check_connection():
            logger.error("Cannot connect to Elasticsearch")
            return
        
        # Create index
        if es_setup.create_index(force_recreate=True):
            logger.info("Index created successfully")
        
        # Load sample data if available
        sample_data_path = "data/preprocessed/sample_businesses.json"
        if os.path.exists(sample_data_path):
            count = es_setup.load_data_from_file(sample_data_path)
            logger.info(f"Loaded {count} businesses into index")
        
        # Get index stats
        stats = es_setup.get_index_stats()
        logger.info(f"Index stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error in setup: {e}")


if __name__ == "__main__":
    main() 