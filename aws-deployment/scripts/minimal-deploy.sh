#!/bin/bash
INSTANCE_TYPE="t3.medium"
AMI_ID="ami-0c02fb55956c7d316"
KEY_NAME="video-mapping-key"
SECURITY_GROUP="video-mapping-sg"
REGION="us-east-1"

# Launch with 30GB storage for ML dependencies
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --region $REGION \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Minimal Deploy Instance: $INSTANCE_ID"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Public IP: $PUBLIC_IP"
echo "Storage: 30GB (should handle ML dependencies)"
