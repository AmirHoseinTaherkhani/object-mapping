#!/bin/bash
# This script will be run ON the EC2 instance

echo "Installing Docker..."
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

echo "Installing AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

echo "Logging into ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 376914316347.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling and running your container..."
docker pull 376914316347.dkr.ecr.us-east-1.amazonaws.com/video-to-map-demo:latest
docker run -d -p 8501:8501 --name video-mapping-demo 376914316347.dkr.ecr.us-east-1.amazonaws.com/video-to-map-demo:latest

echo "✅ Container deployed!"
echo "Access your demo at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
