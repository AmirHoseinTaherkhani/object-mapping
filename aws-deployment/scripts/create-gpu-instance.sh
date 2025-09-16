#!/bin/bash
set -e

# Configuration for GPU instance
INSTANCE_TYPE="g4dn.xlarge"  # 1 GPU, 4 vCPUs, 16GB RAM (~$0.526/hour)
AMI_ID="ami-0c02fb55956c7d316"  # Amazon Linux 2 with GPU support
KEY_NAME="video-mapping-key"
SECURITY_GROUP="video-mapping-sg"
REGION="us-east-1"

echo "Creating GPU instance for video mapping demo..."
echo "Instance type: $INSTANCE_TYPE (estimated cost: ~$0.53/hour)"

# Create key pair if doesn't exist
aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION 2>/dev/null || \
aws ec2 create-key-pair --key-name $KEY_NAME --region $REGION --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem

# Set proper permissions for key
chmod 400 ~/.ssh/${KEY_NAME}.pem 2>/dev/null || echo "Key file not found locally"

echo "Key pair ready: $KEY_NAME"

echo "Creating security group..."
# Create security group for web access
aws ec2 create-security-group \
    --group-name $SECURITY_GROUP \
    --description "Security group for video mapping demo" \
    --region $REGION 2>/dev/null || echo "Security group already exists"

# Allow HTTP access on port 8501 (Streamlit)
aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || echo "Port 8501 rule already exists"

# Allow SSH access
aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || echo "SSH rule already exists"

echo "Launching GPU instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ GPU Instance Ready!"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Access URL: http://$PUBLIC_IP:8501"
echo ""
echo "Next: SSH into instance and run your container"
echo "SSH command: ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$PUBLIC_IP"
