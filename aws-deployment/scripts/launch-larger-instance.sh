#!/bin/bash
INSTANCE_TYPE="t3.xlarge"  # 4 vCPUs, 16GB RAM, ~$0.17/hour
AMI_ID="ami-0c02fb55956c7d316"
KEY_NAME="video-mapping-key"
SECURITY_GROUP="video-mapping-sg"
REGION="us-east-1"

# Launch with 20GB root volume
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --region $REGION \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "New Instance ID: $INSTANCE_ID"
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "New Public IP: $PUBLIC_IP"
