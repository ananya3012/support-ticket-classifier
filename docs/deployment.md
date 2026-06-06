# AWS Deployment Guide

This guide covers deploying the Ticket Classification API to AWS using EC2 and ECS options.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Option 1: EC2 Deployment](#option-1-ec2-deployment)
3. [Option 2: ECS with Fargate](#option-2-ecs-with-fargate)
4. [Security Configuration](#security-configuration)
5. [Monitoring & Logging](#monitoring--logging)

---

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured locally
- Docker installed for building images
- Trained model (run `python scripts/train.py` first)

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
```

---

## Option 1: EC2 Deployment

### Step 1: Launch EC2 Instance

1. **Go to EC2 Console** → Launch Instance

2. **Choose AMI**: Amazon Linux 2023 or Ubuntu 22.04 LTS

3. **Instance Type**: `t3.small` (minimum) or `t3.medium` (recommended)
   - 2 vCPU, 2-4 GB RAM for ML inference

4. **Key Pair**: Create or select existing key pair for SSH access

5. **Network Settings**:
   - Allow SSH (port 22) from your IP
   - Allow HTTP (port 80) from anywhere
   - Allow Custom TCP (port 8000) from anywhere

6. **Storage**: 20 GB gp3 SSD minimum

### Step 2: Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<public-ip>

# Update system
sudo yum update -y  # Amazon Linux
# OR
sudo apt update && sudo apt upgrade -y  # Ubuntu

# Install Docker
sudo yum install docker -y  # Amazon Linux
# OR
sudo apt install docker.io -y  # Ubuntu

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Logout and login again for group changes
exit
ssh -i your-key.pem ec2-user@<public-ip>
```

### Step 3: Deploy Application

```bash
# Clone repository
git clone https://github.com/yourusername/ticket-classifier.git
cd ticket-classifier

# Build Docker image
docker build -t ticket-classifier:latest .

# Run container
docker run -d \
  --name ticket-api \
  -p 8000:8000 \
  --restart unless-stopped \
  ticket-classifier:latest

# Verify deployment
curl http://localhost:8000/health
```

### Step 4: Setup Nginx Reverse Proxy (Optional)

```bash
# Install Nginx
sudo yum install nginx -y  # Amazon Linux
# OR
sudo apt install nginx -y  # Ubuntu

# Configure Nginx
sudo tee /etc/nginx/conf.d/ticket-api.conf << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

---

## Option 2: ECS with Fargate

### Step 1: Push Image to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name ticket-classifier --region us-east-1

# Get login command
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
docker build -t ticket-classifier .
docker tag ticket-classifier:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ticket-classifier:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ticket-classifier:latest
```

### Step 2: Create ECS Cluster

1. **Go to ECS Console** → Clusters → Create Cluster

2. **Cluster name**: `ticket-classifier-cluster`

3. **Infrastructure**: AWS Fargate (Serverless)

4. Click **Create**

### Step 3: Create Task Definition

1. **Go to Task Definitions** → Create new task definition

2. **Task definition family**: `ticket-classifier-task`

3. **Launch type**: AWS Fargate

4. **CPU**: 0.5 vCPU, **Memory**: 1 GB (minimum)

5. **Container definitions**:
   - Name: `ticket-classifier`
   - Image URI: `<account-id>.dkr.ecr.us-east-1.amazonaws.com/ticket-classifier:latest`
   - Port mappings: 8000

### Step 4: Create Service

1. **Go to Clusters** → Select your cluster → Services → Create

2. **Launch type**: Fargate

3. **Task definition**: Select `ticket-classifier-task`

4. **Service name**: `ticket-classifier-service`

5. **Desired tasks**: 1 (or more for high availability)

6. **Networking**:
   - VPC: Select default or your VPC
   - Subnets: Select public subnets
   - Security group: Allow port 8000

7. **Load Balancer** (optional but recommended):
   - Application Load Balancer
   - Health check path: `/health`

---

## Security Configuration

### Security Group Rules

| Type | Protocol | Port | Source | Description |
|------|----------|------|--------|-------------|
| SSH | TCP | 22 | Your IP | Admin access |
| HTTP | TCP | 80 | 0.0.0.0/0 | Public web |
| HTTPS | TCP | 443 | 0.0.0.0/0 | Secure web |
| Custom TCP | TCP | 8000 | 0.0.0.0/0 | API direct |

### SSL/TLS with Let's Encrypt (EC2)

```bash
# Install Certbot
sudo yum install certbot python3-certbot-nginx -y

# Get certificate (replace with your domain)
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Environment Variables

For sensitive configuration, use AWS Secrets Manager or SSM Parameter Store:

```bash
# Create secret
aws secretsmanager create-secret \
  --name ticket-classifier/config \
  --secret-string '{"API_KEY":"your-key","LOG_LEVEL":"info"}'
```

---

## Monitoring & Logging

### CloudWatch Logs (ECS)

Logs are automatically sent to CloudWatch when using ECS Fargate.

```bash
# View logs
aws logs get-log-events \
  --log-group-name /ecs/ticket-classifier-task \
  --log-stream-name <stream-name>
```

### CloudWatch Alarms

```bash
# Create CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name ticket-classifier-high-cpu \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

### Health Monitoring

```bash
# Simple health check script
#!/bin/bash
while true; do
  response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
  if [ "$response" != "200" ]; then
    echo "API unhealthy! Status: $response"
    # Add alerting logic here
  fi
  sleep 60
done
```

---

## Cost Estimation

### EC2 (t3.small, 24/7)
- Instance: ~$15/month
- Storage (20GB): ~$2/month
- **Total**: ~$17/month

### ECS Fargate (0.5 vCPU, 1GB, 24/7)
- Compute: ~$15/month
- Storage: ~$2/month
- Load Balancer (optional): ~$16/month
- **Total**: ~$17-33/month

---

## Quick Start Commands

```bash
# EC2 Quick Deploy
ssh -i key.pem ec2-user@<ip>
git clone <repo> && cd ticket-classifier
docker build -t ticket-classifier .
docker run -d -p 8000:8000 ticket-classifier

# Test API
curl http://<public-ip>:8000/health
curl -X POST http://<public-ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject": "Test", "description": "This is a test ticket"}'
```
