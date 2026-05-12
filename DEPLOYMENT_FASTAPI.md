# Agentic Logistics Optimizer - Production Deployment Guide

## Table of Contents
1. [Quick Start (Local)](#quick-start-local)
2. [Docker Deployment](#docker-deployment)
3. [AWS Lambda Deployment](#aws-lambda-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Monitoring & Debugging](#monitoring--debugging)

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- `pip` package manager
- GitHub Models API token (from GitHub)
- Optional: Redis (for caching)
- Optional: Docker & Docker Compose

### Installation

```bash
# Clone or navigate to project
cd f:\python\AgenticAI

# Create virtual environment
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.production .env

# Edit .env and add:
# GITHUB_TOKEN=your_github_token
# ENVIRONMENT=development
```

### Run FastAPI Server

```bash
# Development (auto-reload)
python -m uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Production
python api_server.py
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# API documentation
# Open in browser: http://localhost:8000/docs

# Test analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "predicted_days": 8.5,
    "promised_days": 7,
    "distance_km": 450,
    "weight_g": 1200,
    "freight_value": 45.0
  }'
```

---

## Docker Deployment

### Build & Run

```bash
# Build image
docker build -t agentic-logistics:latest .

# Run container (standalone)
docker run -d \
  --name agentic-logistics-api \
  -p 8000:8000 \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -e ENVIRONMENT=production \
  agentic-logistics:latest

# Check logs
docker logs -f agentic-logistics-api
```

### Docker Compose (Recommended)

```bash
# Start all services (API + Redis)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all
docker-compose down

# Run with specific environment
ENVIRONMENT=production docker-compose up -d
```

### Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check status
curl http://localhost:8000/status

# Access Swagger UI
# Browser: http://localhost:8000/docs
```

---

## AWS Lambda Deployment

### Prerequisites
- AWS Account with appropriate permissions
- AWS CLI configured (`aws configure`)
- SAM CLI (`pip install aws-sam-cli`)
- Docker (for SAM builds)

### Deployment Steps

#### 1. Create Secrets Manager Entry

```bash
# Store GitHub token securely
aws secretsmanager create-secret \
  --name /agentic-logistics/production/github-token \
  --secret-string '{"GITHUB_TOKEN":"your_github_token_here"}' \
  --region eu-central-1
```

#### 2. Build & Deploy with SAM

```bash
# Build SAM application
sam build

# Deploy (first time - guided)
sam deploy --guided

# Deploy (subsequent times)
sam deploy

# Check stack status
aws cloudformation describe-stacks \
  --stack-name agentic-logistics-api-production \
  --query 'Stacks[0].StackStatus'
```

#### 3. Get API Endpoint

```bash
# Retrieve API Gateway URL
aws cloudformation describe-stacks \
  --stack-name agentic-logistics-api-production \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
  --output text
```

#### 4. Test Lambda Deployment

```bash
# Get API endpoint
API_URL=$(aws cloudformation describe-stacks \
  --stack-name agentic-logistics-api-production \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiEndpoint`].OutputValue' \
  --output text)

# Health check
curl $API_URL/health

# Test analysis
curl -X POST $API_URL/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "predicted_days": 8.5,
    "promised_days": 7,
    "distance_km": 450,
    "weight_g": 1200,
    "freight_value": 45.0
  }'
```

#### 5. Update Lambda Function

```bash
# If you have local changes
sam build
sam deploy  # without --guided

# Or directly update code
aws lambda update-function-code \
  --function-name agentic-logistics-api-production \
  --zip-file fileb://lambda_handler.zip
```

### Lambda Configuration

Key settings in `template.yaml`:

```yaml
MemorySize: 3008        # 3 GB (enough for models)
Timeout: 30             # 30 seconds (adjust as needed)
EphemeralStorage: 10240 # 10 GB for large model files
```

### Cost Optimization

```bash
# Monitor Lambda costs
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=agentic-logistics-api-production \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-31T23:59:59Z \
  --period 86400 \
  --statistics Average,Maximum

# Enable Lambda Insights for detailed metrics
aws lambda update-function-configuration \
  --function-name agentic-logistics-api-production \
  --layers arn:aws:lambda:eu-central-1:580254703988:layer:LambdaInsightsExtension:21
```

---

## Environment Configuration

### Local Development (.env)

```bash
ENVIRONMENT=development
LOG_LEVEL=debug
GITHUB_TOKEN=your_token
GITHUB_MODEL=gpt-4o-mini
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Production (.env.production)

```bash
ENVIRONMENT=production
LOG_LEVEL=info
GITHUB_TOKEN=${AWS_SECRETSMANAGER_GITHUB_TOKEN}
GITHUB_MODEL=gpt-4o-mini
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com
REDIS_HOST=elasticache-endpoint.eu-central-1.cache.amazonaws.com
REDIS_PORT=6379
AWS_REGION=eu-central-1
DYNAMODB_TABLE=agentic-logistics-decisions-production
S3_BUCKET=agentic-logistics-results-[ACCOUNT_ID]-eu-central-1
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

---

## Monitoring & Debugging

### CloudWatch Logs

```bash
# View Lambda logs
aws logs tail /aws/lambda/agentic-logistics-api-production --follow

# View API Gateway logs
aws logs tail /aws/apigateway/agentic-logistics-production --follow

# View specific request
aws logs filter-log-events \
  --log-group-name /aws/lambda/agentic-logistics-api-production \
  --query-string 'request_id="abc12345"'
```

### Metrics & Alarms

```bash
# List alarms
aws cloudwatch describe-alarms \
  --alarm-name-prefix agentic-logistics

# Create custom alarm
aws cloudwatch put-metric-alarm \
  --alarm-name agentic-logistics-high-errors \
  --alarm-description "Alert on error spike" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanOrEqualToThreshold
```

### X-Ray Tracing

```bash
# View service map
aws xray get-service-graph \
  --start-time $(date -u -d '1 hour ago' +%s) \
  --end-time $(date -u +%s)

# Get trace summary
aws xray batch-get-traces \
  --trace-ids $(aws xray get-trace-summaries \
    --start-time $(date -u -d '1 hour ago' +%s) \
    --end-time $(date -u +%s) \
    --query 'TraceSummaries[0].Id' \
    --output text)
```

### Local Debugging

```bash
# Run with debug logs
DEBUG=true LOG_LEVEL=debug python api_server.py

# Test with sample request (Python)
python -c "
from pydantic_agents import DeliveryScenario, run_multi_agent_analysis_parallel
scenario = DeliveryScenario(
    predicted_days=8.5, promised_days=7, distance_km=450,
    weight_g=1200, freight_value=45.0, rag_context='test'
)
result = run_multi_agent_analysis_parallel(scenario)
print(result)
"
```

### Performance Profiling

```bash
# Profile with Python cProfile
python -m cProfile -s cumtime api_server.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler api_server.py
```

---

## Troubleshooting

### Lambda Timeout

**Problem:** "Task timed out after 30 seconds"

**Solution:**
- Increase `Timeout` in `template.yaml`
- Optimize LLM calls (use caching)
- Consider async processing

### High Memory Usage

**Problem:** "Container did not exit cleanly"

**Solution:**
- Increase `MemorySize` in `template.yaml`
- Enable Lambda memory optimization
- Use DynamoDB Streams for batch processing

### Cold Start Issues

**Problem:** First request takes 5-10 seconds

**Solution:**
```bash
# Configure provisioned concurrency
aws lambda put-provisioned-concurrency-config \
  --function-name agentic-logistics-api-production \
  --provisioned-concurrent-executions 5

# Or use Scheduled Lambda Warmer
# (see AWS docs for CloudWatch Events rule)
```

---

## Rollback

```bash
# List previous versions
aws lambda list-versions-by-function \
  --function-name agentic-logistics-api-production

# Rollback to previous version
aws lambda update-alias \
  --function-name agentic-logistics-api-production \
  --name live \
  --function-version N  # Previous version number
```

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [SAM Developer Guide](https://docs.aws.amazon.com/serverless-application-model/)
- [CloudWatch Logs Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AnalyzingLogData.html)

---

## Contact & Support

For issues or questions:
- Check CloudWatch Logs (start here!)
- Review X-Ray traces for errors
- Check Lambda cold start metrics
- Verify environment variables are set correctly
