"""
Elasticsearch Search Module for LCF Group Funding Recommendation System

Handles:
- Semantic search using dense vectors
- Hybrid search (text + vector)
- Business similarity matching
- Advanced filtering and aggregation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError, NotFoundError
from sentence_transformers import SentenceTransformer
from loguru import logger


class BusinessSearch:
    """Business search and similarity matching class"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 9200, 
                 username: str = "elastic", 
                 password: str = "changeme",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        
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
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_name = "lcf_businesses"
        
        logger.info(f"Initialized BusinessSearch for {host}:{port}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def semantic_search(self, 
                       query: str, 
                       limit: int = 10, 
                       min_score: float = 0.1) -> List[Dict[str, Any]]:
        """Perform semantic search using dense vectors"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Build search query
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": limit,
                "_source": [
                    "business_name", "domain", "location", "description", 
                    "features", "risk_assessment", "funding_recommendation",
                    "overall_confidence", "timestamp"
                ]
            }
            
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                if hit['_score'] >= min_score:
                    result = {
                        'id': hit['_id'],
                        'score': hit['_score'],
                        'business_name': hit['_source'].get('business_name', ''),
                        'domain': hit['_source'].get('domain', ''),
                        'location': hit['_source'].get('location', ''),
                        'description': hit['_source'].get('description', ''),
                        'features': hit['_source'].get('features', {}),
                        'risk_assessment': hit['_source'].get('risk_assessment', {}),
                        'funding_recommendation': hit['_source'].get('funding_recommendation', {}),
                        'overall_confidence': hit['_source'].get('overall_confidence', 0.0),
                        'timestamp': hit['_source'].get('timestamp', '')
                    }
                    results.append(result)
            
            logger.info(f"Semantic search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def hybrid_search(self, 
                     query: str, 
                     limit: int = 10, 
                     vector_weight: float = 0.7,
                     text_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Perform hybrid search combining text and vector similarity"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            # Build hybrid search query
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": f"{vector_weight} * cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding}
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["business_name^2", "description", "domain", "location"],
                                    "type": "best_fields",
                                    "boost": text_weight
                                }
                            }
                        ]
                    }
                },
                "size": limit,
                "_source": [
                    "business_name", "domain", "location", "description", 
                    "features", "risk_assessment", "funding_recommendation",
                    "overall_confidence", "timestamp"
                ]
            }
            
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'business_name': hit['_source'].get('business_name', ''),
                    'domain': hit['_source'].get('domain', ''),
                    'location': hit['_source'].get('location', ''),
                    'description': hit['_source'].get('description', ''),
                    'features': hit['_source'].get('features', {}),
                    'risk_assessment': hit['_source'].get('risk_assessment', {}),
                    'funding_recommendation': hit['_source'].get('funding_recommendation', {}),
                    'overall_confidence': hit['_source'].get('overall_confidence', 0.0),
                    'timestamp': hit['_source'].get('timestamp', '')
                }
                results.append(result)
            
            logger.info(f"Hybrid search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def find_similar_businesses(self, 
                              business_data: Dict[str, Any], 
                              limit: int = 10,
                              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find businesses similar to the given business"""
        try:
            # Extract description for similarity search
            description = business_data.get('description', '')
            if not description:
                description = f"{business_data.get('business_name', '')} {business_data.get('domain', '')}"
            
            # Generate embedding
            embedding = self.generate_embedding(description)
            if not embedding:
                return []
            
            # Build search query with filters
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": embedding}
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": limit,
                "_source": [
                    "business_name", "domain", "location", "description", 
                    "features", "risk_assessment", "funding_recommendation",
                    "overall_confidence", "timestamp"
                ]
            }
            
            # Add filters if provided
            if filters:
                filter_queries = []
                
                if 'domain' in filters:
                    filter_queries.append({"term": {"domain": filters['domain']}})
                
                if 'location' in filters:
                    filter_queries.append({"term": {"location": filters['location']}})
                
                if 'risk_level' in filters:
                    filter_queries.append({"term": {"risk_assessment.level": filters['risk_level']}})
                
                if 'funding_decision' in filters:
                    filter_queries.append({"term": {"funding_recommendation.decision": filters['funding_decision']}})
                
                if 'min_revenue' in filters:
                    filter_queries.append({"range": {"features.revenue": {"gte": filters['min_revenue']}}})
                
                if 'max_revenue' in filters:
                    filter_queries.append({"range": {"features.revenue": {"lte": filters['max_revenue']}}})
                
                if filter_queries:
                    search_body["query"]["bool"]["filter"] = filter_queries
            
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_id'],
                    'similarity_score': hit['_score'],
                    'business_name': hit['_source'].get('business_name', ''),
                    'domain': hit['_source'].get('domain', ''),
                    'location': hit['_source'].get('location', ''),
                    'description': hit['_source'].get('description', ''),
                    'features': hit['_source'].get('features', {}),
                    'risk_assessment': hit['_source'].get('risk_assessment', {}),
                    'funding_recommendation': hit['_source'].get('funding_recommendation', {}),
                    'overall_confidence': hit['_source'].get('overall_confidence', 0.0),
                    'timestamp': hit['_source'].get('timestamp', '')
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar businesses")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar businesses: {e}")
            return []
    
    def advanced_search(self, 
                       query: str = "",
                       filters: Optional[Dict[str, Any]] = None,
                       sort_by: str = "relevance",
                       limit: int = 20,
                       offset: int = 0) -> Dict[str, Any]:
        """Advanced search with multiple filters and sorting options"""
        try:
            # Build base query
            search_body = {
                "query": {"bool": {"must": []}},
                "size": limit,
                "from": offset,
                "_source": [
                    "business_name", "domain", "location", "description", 
                    "features", "risk_assessment", "funding_recommendation",
                    "overall_confidence", "timestamp"
                ]
            }
            
            # Add text search if query provided
            if query:
                # Generate embedding for semantic search
                query_embedding = self.generate_embedding(query)
                if query_embedding:
                    search_body["query"]["bool"]["should"] = [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["business_name^2", "description", "domain", "location"],
                                "type": "best_fields"
                            }
                        }
                    ]
                    search_body["query"]["bool"]["minimum_should_match"] = 1
            
            # Add filters
            if filters:
                filter_queries = []
                
                # Domain filter
                if 'domain' in filters and filters['domain']:
                    filter_queries.append({"terms": {"domain": filters['domain'] if isinstance(filters['domain'], list) else [filters['domain']]}})
                
                # Location filter
                if 'location' in filters and filters['location']:
                    filter_queries.append({"terms": {"location": filters['location'] if isinstance(filters['location'], list) else [filters['location']]}})
                
                # Risk level filter
                if 'risk_level' in filters and filters['risk_level']:
                    filter_queries.append({"terms": {"risk_assessment.level": filters['risk_level'] if isinstance(filters['risk_level'], list) else [filters['risk_level']]}})
                
                # Funding decision filter
                if 'funding_decision' in filters and filters['funding_decision']:
                    filter_queries.append({"terms": {"funding_recommendation.decision": filters['funding_decision'] if isinstance(filters['funding_decision'], list) else [filters['funding_decision']]}})
                
                # Revenue range filter
                if 'revenue_min' in filters or 'revenue_max' in filters:
                    revenue_filter = {"range": {"features.revenue": {}}}
                    if 'revenue_min' in filters:
                        revenue_filter["range"]["features.revenue"]["gte"] = filters['revenue_min']
                    if 'revenue_max' in filters:
                        revenue_filter["range"]["features.revenue"]["lte"] = filters['revenue_max']
                    filter_queries.append(revenue_filter)
                
                # Employee count range filter
                if 'employee_min' in filters or 'employee_max' in filters:
                    employee_filter = {"range": {"features.employee_count": {}}}
                    if 'employee_min' in filters:
                        employee_filter["range"]["features.employee_count"]["gte"] = filters['employee_min']
                    if 'employee_max' in filters:
                        employee_filter["range"]["features.employee_count"]["lte"] = filters['employee_max']
                    filter_queries.append(employee_filter)
                
                if filter_queries:
                    search_body["query"]["bool"]["filter"] = filter_queries
            
            # Add sorting
            if sort_by == "revenue":
                search_body["sort"] = [{"features.revenue": {"order": "desc"}}]
            elif sort_by == "employee_count":
                search_body["sort"] = [{"features.employee_count": {"order": "desc"}}]
            elif sort_by == "confidence":
                search_body["sort"] = [{"overall_confidence": {"order": "desc"}}]
            elif sort_by == "timestamp":
                search_body["sort"] = [{"timestamp": {"order": "desc"}}]
            # Default is relevance (score)
            
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'business_name': hit['_source'].get('business_name', ''),
                    'domain': hit['_source'].get('domain', ''),
                    'location': hit['_source'].get('location', ''),
                    'description': hit['_source'].get('description', ''),
                    'features': hit['_source'].get('features', {}),
                    'risk_assessment': hit['_source'].get('risk_assessment', {}),
                    'funding_recommendation': hit['_source'].get('funding_recommendation', {}),
                    'overall_confidence': hit['_source'].get('overall_confidence', 0.0),
                    'timestamp': hit['_source'].get('timestamp', '')
                }
                results.append(result)
            
            # Build response
            search_response = {
                'results': results,
                'total': response['hits']['total']['value'],
                'query': query,
                'filters': filters,
                'sort_by': sort_by,
                'limit': limit,
                'offset': offset
            }
            
            logger.info(f"Advanced search returned {len(results)} results out of {search_response['total']} total")
            return search_response
            
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            return {'results': [], 'total': 0, 'error': str(e)}
    
    def get_aggregations(self, field: str, size: int = 20) -> Dict[str, Any]:
        """Get aggregations for a specific field"""
        try:
            search_body = {
                "size": 0,
                "aggs": {
                    "field_agg": {
                        "terms": {
                            "field": field,
                            "size": size
                        }
                    }
                }
            }
            
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            buckets = response['aggregations']['field_agg']['buckets']
            return {
                'field': field,
                'buckets': buckets,
                'total': len(buckets)
            }
            
        except Exception as e:
            logger.error(f"Error getting aggregations for {field}: {e}")
            return {'field': field, 'buckets': [], 'total': 0}
    
    def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on business names and domains"""
        try:
            search_body = {
                "suggest": {
                    "business_suggestions": {
                        "prefix": query,
                        "completion": {
                            "field": "business_name_suggest",
                            "size": limit
                        }
                    },
                    "domain_suggestions": {
                        "prefix": query,
                        "completion": {
                            "field": "domain_suggest",
                            "size": limit
                        }
                    }
                }
            }
            
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            suggestions = set()
            
            # Extract business name suggestions
            if 'business_suggestions' in response['suggest']:
                for suggestion in response['suggest']['business_suggestions']:
                    for option in suggestion['options']:
                        suggestions.add(option['text'])
            
            # Extract domain suggestions
            if 'domain_suggestions' in response['suggest']:
                for suggestion in response['suggest']['domain_suggestions']:
                    for option in suggestion['options']:
                        suggestions.add(option['text'])
            
            return list(suggestions)[:limit]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []


def main():
    """Example usage of the search module"""
    # Initialize search
    search = BusinessSearch(
        host="localhost",
        port=9200,
        username="elastic",
        password="changeme"
    )
    
    try:
        # Semantic search example
        print("=== Semantic Search ===")
        results = search.semantic_search("AI technology startup", limit=5)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['business_name']} ({result['domain']}) - Score: {result['score']:.3f}")
        
        # Hybrid search example
        print("\n=== Hybrid Search ===")
        results = search.hybrid_search("California tech companies", limit=5)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['business_name']} ({result['location']}) - Score: {result['score']:.3f}")
        
        # Advanced search example
        print("\n=== Advanced Search ===")
        filters = {
            'domain': 'Technology',
            'risk_level': 'medium',
            'revenue_min': 100000
        }
        results = search.advanced_search(
            query="SaaS platform",
            filters=filters,
            sort_by="revenue",
            limit=5
        )
        for i, result in enumerate(results['results'], 1):
            revenue = result['features'].get('revenue', 0)
            try:
                formatted_revenue = f"${revenue:,}"
                print(f"{i}. {result['business_name']} - Revenue: {formatted_revenue}")
            except (ValueError, TypeError):
                print(f"{i}. {result['business_name']} - Revenue: ${revenue}")
        
        # Get aggregations
        print("\n=== Domain Aggregations ===")
        domain_agg = search.get_aggregations("domain")
        for bucket in domain_agg['buckets']:
            print(f"{bucket['key']}: {bucket['doc_count']} businesses")
        
    except Exception as e:
        logger.error(f"Error in search example: {e}")


if __name__ == "__main__":
    main() 