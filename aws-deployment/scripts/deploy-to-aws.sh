#!/bin/bash
set -e

# Configuration
REGION="us-east-1"  # Change if needed
ECR_REPO_NAME="video-to-map-demo"
CLUSTER_NAME="video-mapping-cluster"
SERVICE_NAME="video-mapping-service"

echo "Starting AWS deployment process..."
echo "Region: $REGION"
echo "ECR Repository: $ECR_REPO_NAME"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: $ACCOUNT_ID"

# ECR repository URI
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME"
echo "ECR URI: $ECR_URI"

echo "Step 1: Creating ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $REGION 2>/dev/null || \
aws ecr create-repository --repository-name $ECR_REPO_NAME --region $REGION

echo "Step 2: Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

echo "Step 3: Building and tagging Docker image..."
docker build -f Dockerfile.alternative -t $ECR_REPO_NAME .
docker tag $ECR_REPO_NAME:latest $ECR_URI:latest

echo "Step 4: Pushing image to ECR..."
docker push $ECR_URI:latest

echo "Image pushed successfully to: $ECR_URI:latest"
echo "Image size and details:"
aws ecr describe-images --repository-name $ECR_REPO_NAME --region $REGION
