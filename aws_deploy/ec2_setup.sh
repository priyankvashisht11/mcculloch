#!/bin/bash

# LCF Group Funding Recommendation System - EC2 Setup Script
# This script sets up an EC2 instance for deploying the funding recommendation system

set -e

# Configuration
INSTANCE_TYPE="t3.large"
AMI_ID="ami-0c02fb55956c7d316"  # Amazon Linux 2 AMI (us-east-1)
KEY_NAME="lcf-funding-key"
SECURITY_GROUP_NAME="lcf-funding-sg"
S3_BUCKET="lcf-funding-models"
REGION="us-east-1"
VOLUME_SIZE=50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if AWS CLI is installed
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install it first."
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI is not configured. Please run 'aws configure' first."
    fi
    
    log "AWS CLI is properly configured"
}

# Create S3 bucket for model storage
create_s3_bucket() {
    log "Creating S3 bucket: $S3_BUCKET"
    
    if aws s3 ls "s3://$S3_BUCKET" 2>&1 | grep -q 'NoSuchBucket'; then
        aws s3 mb "s3://$S3_BUCKET" --region $REGION
        aws s3api put-bucket-versioning --bucket $S3_BUCKET --versioning-configuration Status=Enabled
        log "S3 bucket created successfully"
    else
        log "S3 bucket already exists"
    fi
}

# Create security group
create_security_group() {
    log "Creating security group: $SECURITY_GROUP_NAME"
    
    # Check if security group exists
    if aws ec2 describe-security-groups --group-names $SECURITY_GROUP_NAME --region $REGION 2>&1 | grep -q 'InvalidGroup.NotFound'; then
        # Create security group
        SG_ID=$(aws ec2 create-security-group \
            --group-name $SECURITY_GROUP_NAME \
            --description "Security group for LCF Funding Recommendation System" \
            --region $REGION \
            --query 'GroupId' --output text)
        
        # Add inbound rules
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 443 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 8000 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 9200 \
            --cidr 0.0.0.0/0 \
            --region $REGION
        
        log "Security group created with ID: $SG_ID"
    else
        SG_ID=$(aws ec2 describe-security-groups \
            --group-names $SECURITY_GROUP_NAME \
            --region $REGION \
            --query 'SecurityGroups[0].GroupId' --output text)
        log "Security group already exists with ID: $SG_ID"
    fi
}

# Create EC2 instance
create_ec2_instance() {
    log "Creating EC2 instance..."
    
    # Get security group ID
    SG_ID=$(aws ec2 describe-security-groups \
        --group-names $SECURITY_GROUP_NAME \
        --region $REGION \
        --query 'SecurityGroups[0].GroupId' --output text)
    
    # Create instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=LCF-Funding-System},{Key=Project,Value=LCF-Funding}]" \
        --region $REGION \
        --query 'Instances[0].InstanceId' --output text)
    
    log "EC2 instance created with ID: $INSTANCE_ID"
    
    # Wait for instance to be running
    log "Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region $REGION \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    
    log "Instance is running with public IP: $PUBLIC_IP"
    
    # Save instance details
    echo "INSTANCE_ID=$INSTANCE_ID" > instance_details.txt
    echo "PUBLIC_IP=$PUBLIC_IP" >> instance_details.txt
    echo "SG_ID=$SG_ID" >> instance_details.txt
}

# Create user data script for instance setup
create_user_data() {
    cat > user_data.sh << 'EOF'
#!/bin/bash

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
yum install -y git

# Create application directory
mkdir -p /opt/lcf-funding
cd /opt/lcf-funding

# Clone repository (replace with your actual repository URL)
# git clone https://github.com/your-org/lcf-funding-ai.git .

# Create environment file
cat > .env << 'ENVEOF'
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
ELASTICSEARCH_PASSWORD=changeme
MODEL_S3_BUCKET=lcf-funding-models
ENVEOF

# Create docker-compose override for production
cat > docker-compose.prod.yml << 'COMPOSEEOF'
version: '3.8'

services:
  api:
    environment:
      - LOG_LEVEL=INFO
      - ELASTICSEARCH_PASSWORD=changeme
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  elasticsearch:
    environment:
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
COMPOSEEOF

# Start services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Create systemd service for auto-start
cat > /etc/systemd/system/lcf-funding.service << 'SERVICEEOF'
[Unit]
Description=LCF Funding Recommendation System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/lcf-funding
ExecStart=/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl enable lcf-funding.service

# Create monitoring script
cat > /opt/lcf-funding/monitor.sh << 'MONITOREOF'
#!/bin/bash

# Check if services are running
if ! docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps | grep -q "Up"; then
    echo "$(date): Services are down, restarting..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml restart
fi

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "$(date): Disk usage is high: ${DISK_USAGE}%"
fi

# Check memory usage
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.2f", $3*100/$2}')
if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "$(date): Memory usage is high: ${MEMORY_USAGE}%"
fi
MONITOREOF

chmod +x /opt/lcf-funding/monitor.sh

# Add monitoring to crontab
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/lcf-funding/monitor.sh >> /var/log/lcf-monitor.log 2>&1") | crontab -

# Create log rotation
cat > /etc/logrotate.d/lcf-funding << 'LOGROTATEEOF'
/var/log/lcf-monitor.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 644 root root
}
LOGROTATEEOF

echo "Setup completed successfully!"
EOF

    log "User data script created"
}

# Deploy application
deploy_application() {
    log "Deploying application to EC2 instance..."
    
    # Read instance details
    source instance_details.txt
    
    # Wait for SSH to be available
    log "Waiting for SSH to be available..."
    while ! nc -z $PUBLIC_IP 22; do
        sleep 5
    done
    
    # Copy files to instance
    log "Copying application files to instance..."
    scp -i ~/.ssh/$KEY_NAME.pem -r . ec2-user@$PUBLIC_IP:/opt/lcf-funding/
    
    # Execute setup commands
    log "Executing setup commands on instance..."
    ssh -i ~/.ssh/$KEY_NAME.pem ec2-user@$PUBLIC_IP << 'EOF'
        cd /opt/lcf-funding
        chmod +x user_data.sh
        ./user_data.sh
EOF
    
    log "Application deployed successfully!"
    log "Access the API at: http://$PUBLIC_IP:8000"
    log "API Documentation: http://$PUBLIC_IP:8000/docs"
    log "Elasticsearch: http://$PUBLIC_IP:9200"
}

# Create CloudWatch dashboard
create_cloudwatch_dashboard() {
    log "Creating CloudWatch dashboard..."
    
    source instance_details.txt
    
    # Create dashboard JSON
    cat > dashboard.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/EC2", "CPUUtilization", "InstanceId", "$INSTANCE_ID" ]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "EC2 CPU Utilization",
                "period": 300
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/EC2", "NetworkIn", "InstanceId", "$INSTANCE_ID" ],
                    [ ".", "NetworkOut", ".", "." ]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Network Traffic",
                "period": 300
            }
        }
    ]
}
EOF
    
    # Create dashboard
    aws cloudwatch put-dashboard \
        --dashboard-name "LCF-Funding-System" \
        --dashboard-body file://dashboard.json \
        --region $REGION
    
    log "CloudWatch dashboard created"
}

# Main execution
main() {
    log "Starting LCF Funding Recommendation System deployment..."
    
    check_aws_cli
    create_s3_bucket
    create_security_group
    create_user_data
    create_ec2_instance
    deploy_application
    create_cloudwatch_dashboard
    
    log "Deployment completed successfully!"
    log "Instance details saved in instance_details.txt"
    
    # Display final information
    source instance_details.txt
    echo ""
    echo "=== DEPLOYMENT SUMMARY ==="
    echo "Instance ID: $INSTANCE_ID"
    echo "Public IP: $PUBLIC_IP"
    echo "Security Group: $SG_ID"
    echo "S3 Bucket: $S3_BUCKET"
    echo ""
    echo "=== ACCESS INFORMATION ==="
    echo "API: http://$PUBLIC_IP:8000"
    echo "API Docs: http://$PUBLIC_IP:8000/docs"
    echo "Elasticsearch: http://$PUBLIC_IP:9200"
    echo "SSH: ssh -i ~/.ssh/$KEY_NAME.pem ec2-user@$PUBLIC_IP"
    echo ""
    echo "=== NEXT STEPS ==="
    echo "1. Update .env file with your AWS credentials"
    echo "2. Upload your trained model to S3: $S3_BUCKET"
    echo "3. Test the API endpoints"
    echo "4. Set up monitoring and alerting"
}

# Run main function
main "$@" 