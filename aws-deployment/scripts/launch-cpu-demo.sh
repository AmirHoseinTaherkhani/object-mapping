#!/bin/bash
INSTANCE_TYPE="t3.large"  # 2 vCPUs, 8GB RAM, ~$0.08/hour
AMI_ID="ami-0c02fb55956c7d316"
KEY_NAME="video-mapping-key"
SECURITY_GROUP="video-mapping-sg"
REGION="us-east-1"

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "CPU Demo Instance ID: $INSTANCE_ID"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ CPU Demo Instance Ready!"
echo "Public IP: $PUBLIC_IP"
echo "Cost: ~$0.08/hour"
