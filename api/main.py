"""
FastAPI Main Application for LCF Group Funding Recommendation System

Provides REST API endpoints for:
- Business profile submission and prediction
- Semantic search of similar businesses
- Model information and health checks
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Import our modules
from model.infer import FundingInference, FundingAnalyzer
from elasticsearch.search import BusinessSearch
from elasticsearch.setup_index import ElasticsearchSetup
from preprocessing.clean_and_embed import BusinessPreprocessor


# Configure logging
logger.add("logs/api.log", rotation="10 MB", level="INFO")


# Pydantic models for API requests/responses
class BusinessProfileRequest(BaseModel):
    """Request model for business profile submission"""
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


class BusinessProfileResponse(BaseModel):
    """Response model for business profile analysis"""
    business_name: str
    risk_assessment: Dict[str, Any]
    funding_recommendation: Dict[str, Any]
    overall_confidence: float
    analysis: Dict[str, Any]
    recommendations: List[str]
    similar_businesses: List[Dict[str, Any]]
    timestamp: str
    model_version: Dict[str, Any]


class SearchRequest(BaseModel):
    """Request model for business search"""
    query: str = Field(..., description="Search query")
    limit: int = Field(10, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    search_type: str = Field("hybrid", description="Search type: semantic, hybrid, or advanced")


class SearchResponse(BaseModel):
    """Response model for business search"""
    results: List[Dict[str, Any]]
    total: int
    query: str
    search_type: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_path: str
    base_model: str
    device: str
    num_labels: int
    max_length: int
    version: Dict[str, Any]
    lora_config: Dict[str, Any]
    risk_labels: List[str]
    funding_labels: List[str]
    total_parameters: Optional[int]
    trainable_parameters: Optional[int]


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]


# Global variables for services
funding_analyzer = None
business_search = None
es_setup = None
preprocessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global funding_analyzer, business_search, es_setup, preprocessor
    
    # Startup
    logger.info("Starting LCF Funding Recommendation API...")
    
    try:
        # Initialize services
        model_path = os.getenv("MODEL_PATH", "models/lora_finetuned")
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        es_port = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
        es_username = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
        es_password = os.getenv("ELASTICSEARCH_PASSWORD", "changeme")
        
        # Initialize funding analyzer
        logger.info("Initializing funding analyzer...")
        funding_analyzer = FundingAnalyzer(model_path)
        
        # Initialize search service
        logger.info("Initializing search service...")
        business_search = BusinessSearch(
            host=es_host,
            port=es_port,
            username=es_username,
            password=es_password
        )
        
        # Initialize Elasticsearch setup
        logger.info("Initializing Elasticsearch setup...")
        es_setup = ElasticsearchSetup(
            host=es_host,
            port=es_port,
            username=es_username,
            password=es_password
        )
        
        # Initialize preprocessor
        logger.info("Initializing preprocessor...")
        preprocessor = BusinessPreprocessor()
        
        logger.info("All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LCF Funding Recommendation API...")


# Create FastAPI app
app = FastAPI(
    title="LCF Group Funding Recommendation API",
    description="AI-powered funding analysis and recommendation system for small businesses",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency functions
def get_funding_analyzer() -> FundingAnalyzer:
    """Get funding analyzer instance"""
    if funding_analyzer is None:
        raise HTTPException(status_code=503, detail="Funding analyzer not initialized")
    return funding_analyzer


def get_business_search() -> BusinessSearch:
    """Get business search instance"""
    if business_search is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    return business_search


def get_es_setup() -> ElasticsearchSetup:
    """Get Elasticsearch setup instance"""
    if es_setup is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not initialized")
    return es_setup


def get_preprocessor() -> BusinessPreprocessor:
    """Get preprocessor instance"""
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not initialized")
    return preprocessor


# API Routes
@app.post("/submit_business_profile", response_model=BusinessProfileResponse)
async def submit_business_profile(
    request: BusinessProfileRequest,
    background_tasks: BackgroundTasks,
    analyzer: FundingAnalyzer = Depends(get_funding_analyzer),
    search: BusinessSearch = Depends(get_business_search),
    es: ElasticsearchSetup = Depends(get_es_setup),
    preproc: BusinessPreprocessor = Depends(get_preprocessor)
):
    """Submit a business profile for funding analysis"""
    try:
        logger.info(f"Processing business profile: {request.business_name}")
        
        # Convert request to dict
        business_data = request.dict()
        
        # Preprocess business data
        preprocessed = preproc.preprocess_business(business_data)
        
        # Analyze business
        analysis_result = analyzer.analyze_business(business_data)
        
        # Find similar businesses
        similar_businesses = search.find_similar_businesses(
            business_data, 
            limit=5,
            filters={"domain": business_data.get("domain")}
        )
        
        # Index the business in background (optional)
        def index_business():
            try:
                # Add preprocessed data to analysis result
                indexed_data = {
                    **analysis_result,
                    "embedding": preprocessed.embedding,
                    "cleaned_description": preprocessed.cleaned_description,
                    "entities": preprocessed.entities,
                    "features": preprocessed.features,
                    "timestamp": datetime.now().isoformat()
                }
                es.index_business(indexed_data)
            except Exception as e:
                logger.error(f"Error indexing business: {e}")
        
        background_tasks.add_task(index_business)
        
        # Prepare response
        response = BusinessProfileResponse(
            business_name=analysis_result["business_name"],
            risk_assessment=analysis_result["risk_assessment"],
            funding_recommendation=analysis_result["funding_recommendation"],
            overall_confidence=analysis_result["overall_confidence"],
            analysis=analysis_result["analysis"],
            recommendations=analysis_result["recommendations"],
            similar_businesses=similar_businesses,
            timestamp=analysis_result["timestamp"],
            model_version=analysis_result["model_version"]
        )
        
        logger.info(f"Successfully analyzed business: {request.business_name}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing business profile: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing business profile: {str(e)}")


@app.get("/search_similar", response_model=SearchResponse)
async def search_similar_businesses(
    query: str,
    limit: int = 10,
    search_type: str = "hybrid",
    filters: Optional[str] = None,
    search: BusinessSearch = Depends(get_business_search)
):
    """Search for similar businesses"""
    try:
        logger.info(f"Searching for businesses similar to: {query}")
        
        # Parse filters if provided
        parsed_filters = None
        if filters:
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid filters format")
        
        # Perform search based on type
        if search_type == "semantic":
            results = search.semantic_search(query, limit=limit)
        elif search_type == "hybrid":
            results = search.hybrid_search(query, limit=limit)
        elif search_type == "advanced":
            search_response = search.advanced_search(
                query=query,
                filters=parsed_filters,
                limit=limit
            )
            results = search_response["results"]
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")
        
        response = SearchResponse(
            results=results,
            total=len(results),
            query=query,
            search_type=search_type,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Search returned {len(results)} results")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


@app.get("/get_model_info", response_model=ModelInfoResponse)
async def get_model_info(
    analyzer: FundingAnalyzer = Depends(get_funding_analyzer)
):
    """Get model information and metadata"""
    try:
        model_info = analyzer.inference.get_model_info()
        
        response = ModelInfoResponse(
            model_path=model_info["model_path"],
            base_model=model_info["base_model"],
            device=model_info["device"],
            num_labels=model_info["num_labels"],
            max_length=model_info["max_length"],
            version=model_info["version"],
            lora_config=model_info["lora_config"],
            risk_labels=model_info["risk_labels"],
            funding_labels=model_info["funding_labels"],
            total_parameters=model_info.get("total_parameters"),
            trainable_parameters=model_info.get("trainable_parameters")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.get("/healthcheck", response_model=HealthCheckResponse)
async def health_check(
    analyzer: FundingAnalyzer = Depends(get_funding_analyzer),
    search: BusinessSearch = Depends(get_business_search),
    es: ElasticsearchSetup = Depends(get_es_setup)
):
    """Check system health and service status"""
    try:
        services = {}
        
        # Check funding analyzer
        try:
            analyzer.inference.get_model_info()
            services["funding_analyzer"] = "healthy"
        except Exception as e:
            services["funding_analyzer"] = f"error: {str(e)}"
        
        # Check search service
        try:
            # Try a simple search
            search.semantic_search("test", limit=1)
            services["search_service"] = "healthy"
        except Exception as e:
            services["search_service"] = f"error: {str(e)}"
        
        # Check Elasticsearch
        try:
            es.check_connection()
            services["elasticsearch"] = "healthy"
        except Exception as e:
            services["elasticsearch"] = f"error: {str(e)}"
        
        # Determine overall status
        overall_status = "healthy" if all("healthy" in status for status in services.values()) else "degraded"
        
        response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            services=services
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing health check: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LCF Group Funding Recommendation API",
        "version": "1.0.0",
        "description": "AI-powered funding analysis and recommendation system",
        "endpoints": {
            "submit_business_profile": "POST /submit_business_profile",
            "search_similar": "GET /search_similar",
            "get_model_info": "GET /get_model_info",
            "healthcheck": "GET /healthcheck",
            "docs": "GET /docs"
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 