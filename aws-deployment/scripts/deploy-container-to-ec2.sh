#!/bin/bash
set -e

# Configuration
REGION="us-east-1"
INSTANCE_ID="i-0a8f9460197fb8a17"
ECR_URI="376914316347.dkr.ecr.us-east-1.amazonaws.com/video-to-map-demo:latest"
KEY_NAME="video-mapping-key"

echo "Getting instance public IP..."
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance Public IP: $PUBLIC_IP"
echo "Instance Status Check..."
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID --region $REGION

echo ""
echo "✅ Ready to deploy! Next steps:"
echo "1. SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$PUBLIC_IP"
echo "2. Install Docker and run your container"
echo "3. Access demo at: http://$PUBLIC_IP:8501"
