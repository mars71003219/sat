# Kubernetes Deployment Guide

## Prerequisites

- Kubernetes cluster with GPU support (NVIDIA device plugin)
- kubectl configured
- Docker images built and available

## Deployment Steps

### 1. Create Secrets

```bash
kubectl apply -f secrets.yaml
```

### 2. Deploy Infrastructure Services

Deploy Redis, PostgreSQL, Kafka first:

```bash
# These would typically come from Helm charts or separate manifests
# Example for Redis:
helm install redis bitnami/redis
helm install postgresql bitnami/postgresql
helm install kafka bitnami/kafka
```

### 3. Deploy AI Service

```bash
kubectl apply -f ai-deployment.yaml
```

### 4. Deploy Celery Workers

```bash
kubectl apply -f celery-worker-deployment.yaml
```

### 5. Verify Deployments

```bash
kubectl get pods
kubectl get services
kubectl get hpa
```

## Scaling

### Manual Scaling

```bash
# Scale AI service
kubectl scale deployment ai-service --replicas=5

# Scale Celery workers
kubectl scale deployment celery-worker --replicas=8
```

### Auto-scaling

The HorizontalPodAutoscaler (HPA) automatically scales Celery workers based on CPU and memory usage (70% and 80% respectively).

## Monitoring

```bash
# View logs
kubectl logs -f deployment/ai-service
kubectl logs -f deployment/celery-worker

# Check health
kubectl port-forward service/ai-service 8000:8000
curl http://localhost:8000/health
```

## GPU Configuration

Ensure NVIDIA device plugin is installed:

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

## Ingress (Optional)

Create an Ingress resource to expose the AI service:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-service-ingress
spec:
  rules:
  - host: ai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-service
            port:
              number: 8000
```

## Clean Up

```bash
kubectl delete -f celery-worker-deployment.yaml
kubectl delete -f ai-deployment.yaml
kubectl delete -f secrets.yaml
```
