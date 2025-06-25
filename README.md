# LCF Group - Intelligent Funding Recommendation System

## 🎯 Overview

An end-to-end AI-based funding analysis system that evaluates small business profiles, assigns funding potential and risk scores using Transformer-based models with LoRA fine-tuning, and enables semantic search of similar past businesses.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │    │  Preprocessing  │    │   AI Model      │
│   (CSV/PDF/API) │───▶│   Pipeline      │───▶│   (LoRA + PEFT) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Elasticsearch │◀───│   FastAPI       │◀───│   Predictions   │
│   (Semantic     │    │   Backend       │    │   & Risk Scores │
│    Search)      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- AWS CLI (for deployment)
- 8GB+ RAM recommended

### Local Development Setup

1. **Clone and Setup**
```bash
git clone <repository-url>
cd lcf_funding_ai
```

2. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Start Services with Docker**
```bash
docker-compose up -d
```

4. **Initialize System**
```bash
# Setup Elasticsearch index
python elasticsearch/setup_index.py

# Train initial model (optional - pre-trained model included)
python model/train_with_lora.py
```

5. **Access Services**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Elasticsearch: http://localhost:9200
- Frontend: http://localhost:8501 (if using Streamlit)

## 📁 Project Structure

```
lcf_funding_ai/
├── data_ingestion/          # Data input handling
├── preprocessing/           # Text cleaning & embedding
├── model/                  # AI model training & inference
├── elasticsearch/          # Search engine setup
├── api/                    # FastAPI backend
├── frontend/               # Streamlit dashboard
├── docker/                 # Containerization
├── aws_deploy/             # AWS deployment scripts
└── tests/                  # Unit tests
```

## 🔧 API Endpoints

### Core Endpoints
- `POST /submit_business_profile` - Submit business profile for analysis
- `GET /search_similar` - Find similar businesses
- `GET /get_model_info` - Model metadata
- `GET /healthcheck` - System health status

### Example Usage
```bash
# Submit business profile
curl -X POST "http://localhost:8000/submit_business_profile" \
  -H "Content-Type: application/json" \
  -d '{
    "business_name": "TechStart Inc",
    "domain": "Technology",
    "years_active": 3,
    "revenue": 500000,
    "employee_count": 15,
    "location": "California",
    "description": "AI-powered SaaS platform"
  }'

# Search similar businesses
curl -X GET "http://localhost:8000/search_similar?query=AI SaaS platform&limit=5"
```

## 🐳 Docker Deployment

### Local Docker Setup
```bash
# Build and start all services
docker-compose up --build

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Production Docker Setup
```bash
# Build production images
docker build -f docker/Dockerfile.api -t lcf-api:latest .
docker build -f docker/Dockerfile.model -t lcf-model:latest .

# Run with production settings
docker-compose -f docker-compose.prod.yml up -d
```

## ☁️ AWS Deployment

### Prerequisites
- AWS CLI configured
- Docker installed on EC2
- S3 bucket for model artifacts

### Quick Deployment
```bash
# Run deployment script
chmod +x aws_deploy/ec2_setup.sh
./aws_deploy/ec2_setup.sh

# Or use Terraform (if available)
cd aws_deploy/terraform
terraform init
terraform apply
```

### Manual EC2 Setup
1. Launch EC2 instance (t3.large or larger)
2. Install Docker and Docker Compose
3. Clone repository
4. Configure environment variables
5. Run `docker-compose up -d`

## 📊 Model Information

### Architecture
- **Base Model**: DistilBERT (bert-base-uncased)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) with PEFT
- **Task**: Multi-class classification (Risk: Low/Medium/High, Funding: Yes/No/Maybe)
- **Input Features**: Business description, sector, financial metrics, location

### Training
```bash
# Train model with custom data
python model/train_with_lora.py --data_path data/training_data.csv --epochs 10

# Evaluate model
python model/evaluate.py --model_path models/lora_finetuned
```

## 🔍 Semantic Search

### Elasticsearch Features
- Dense vector search using sentence embeddings
- Business similarity matching
- Configurable search parameters
- Real-time indexing

### Search Configuration
```python
# Example search query
{
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": embedding_vector}
            }
        }
    }
}
```

## 📈 Monitoring & Logging

### Logging Configuration
- API access logs
- Model prediction logs
- Error tracking
- Performance metrics

### Health Checks
```bash
# Check system health
curl http://localhost:8000/healthcheck

# Monitor logs
docker-compose logs -f api elasticsearch
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# API tests
pytest tests/api/
```

### Test Coverage
```bash
pytest --cov=api --cov=model --cov=elasticsearch tests/
```

## 🔒 Security

### Environment Variables
```bash
# Required environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export ELASTICSEARCH_PASSWORD=your_password
export MODEL_S3_BUCKET=your_bucket
```

### Security Best Practices
- Use AWS IAM roles for EC2
- Enable HTTPS in production
- Implement API rate limiting
- Regular security updates

## 📝 Configuration

### Model Configuration
Edit `model/config/model_config.yaml`:
```yaml
model:
  base_model: "distilbert-base-uncased"
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_lin", "v_lin"]
  training:
    learning_rate: 2e-4
    batch_size: 16
    epochs: 10
```

### API Configuration
Edit `api/config/settings.py`:
```python
ELASTICSEARCH_URL = "http://elasticsearch:9200"
MODEL_PATH = "/app/models/lora_finetuned"
LOG_LEVEL = "INFO"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

This project is proprietary to LCF Group. All rights reserved.

## 🆘 Support

For technical support:
- Create an issue in the repository
- Contact: tech-support@lcfgroup.com
- Documentation: `/docs` directory

## 🔄 Version History

- **v1.0.0** - Initial release with LoRA fine-tuning
- **v1.1.0** - Added semantic search capabilities
- **v1.2.0** - Docker containerization
- **v1.3.0** - AWS deployment support 