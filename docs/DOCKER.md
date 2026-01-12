# Docker Deployment Guide

## Prerequisites
- Docker installed (version 20.10+)
- Docker Compose installed (version 2.0+)

## Building the Docker Image

### Basic Build
```bash
docker build -t customer-churn-api:latest .
```

### Build with Custom Tag
```bash
docker build -t customer-churn-api:v1.0.0 .
```

## Running the Container

### Basic Run
```bash
docker run -p 8000:8000 customer-churn-api:latest
```

### Run with Environment Variables
```bash
docker run -p 8000:8000 \
  -e DEBUG=True \
  -e APP_NAME="Customer Churn API" \
  customer-churn-api:latest
```

### Run with Volume Mounts
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  customer-churn-api:latest
```

## Using Docker Compose

### Start Services
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f
```

### Stop Services
```bash
docker-compose down
```

### Rebuild and Start
```bash
docker-compose up -d --build
```

## Accessing the Application

Once the container is running:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/

## Container Structure

```
/app
├── app/              # FastAPI application
├── ml_pipeline/      # ML training pipeline
├── models/          # Trained models (mounted volume)
├── logs/            # Application logs (mounted volume)
└── Scripts/         # Data files (mounted volume)
```

## Environment Variables

- `DEBUG`: Enable debug mode (default: False)
- `APP_NAME`: Application name (default: Customer Churn Prediction API)
- `PYTHONUNBUFFERED`: Disable Python output buffering (set to 1)

## Troubleshooting

### Port Already in Use
```bash
# Use a different port
docker run -p 8080:8000 customer-churn-api:latest
```

### Permission Issues
```bash
# On Linux, you may need to adjust permissions
sudo chown -R $USER:$USER models/ logs/
```

### View Container Logs
```bash
docker logs <container_id>
docker-compose logs churn-api
```

### Execute Commands in Container
```bash
docker exec -it <container_id> /bin/bash
```

### Check Container Status
```bash
docker ps
docker-compose ps
```

## Production Deployment

For production, consider:
1. Using a reverse proxy (nginx)
2. Setting up SSL/TLS
3. Using environment-specific configurations
4. Implementing health checks
5. Setting resource limits
6. Using orchestration tools (Kubernetes, Docker Swarm)

## Example Production docker-compose.yml

```yaml
version: '3.8'

services:
  churn-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
```

