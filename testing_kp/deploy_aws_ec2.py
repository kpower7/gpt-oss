#!/usr/bin/env python3
"""
AWS EC2 Deployment Script for GPT OSS Weather Demo

This script helps you deploy the GPT OSS weather tool calling demo on AWS EC2
with GPU support using vLLM for OpenAI-compatible API serving.

Requirements:
- AWS CLI configured with your credentials
- boto3 installed
- Your AWS account with GPU instance limits enabled
"""

import boto3
import time
import json
from typing import Dict, Any

# Configuration
AWS_REGION = "us-east-1"  # Change to your preferred region
INSTANCE_TYPE = "g4dn.xlarge"  # T4 GPU, 16GB VRAM, good for gpt-oss-20b
AMI_ID = "ami-0c02fb55956c7d316"  # Amazon Linux 2 AMI (update as needed)
KEY_PAIR_NAME = "gpt-oss-key"  # You'll need to create this
SECURITY_GROUP_NAME = "gpt-oss-sg"

# User data script for automatic setup
USER_DATA_SCRIPT = """#!/bin/bash
set -e

# Update system
yum update -y
yum install -y docker git python3-pip

# Install NVIDIA drivers
yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring-1.0-1.noarch.rpm
rpm -i cuda-keyring-1.0-1.noarch.rpm
yum install -y cuda-drivers

# Install Docker and NVIDIA Container Toolkit
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
yum install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Install Python dependencies
pip3 install --upgrade pip
pip3 install vllm transformers torch openai-harmony requests python-dotenv pytz

# Clone the repository
cd /home/ec2-user
git clone https://github.com/openai/gpt-oss.git
chown -R ec2-user:ec2-user gpt-oss

# Create startup script
cat > /home/ec2-user/start_gpt_oss.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start vLLM server with GPT OSS 20B
python -m vllm.entrypoints.openai.api_server \\
    --model openai/gpt-oss-20b \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 4096
EOF

chmod +x /home/ec2-user/start_gpt_oss.sh
chown ec2-user:ec2-user /home/ec2-user/start_gpt_oss.sh

# Create systemd service for auto-start
cat > /etc/systemd/system/gpt-oss.service << 'EOF'
[Unit]
Description=GPT OSS vLLM Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
ExecStart=/home/ec2-user/start_gpt_oss.sh
Restart=always
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[Install]
WantedBy=multi-user.target
EOF

systemctl enable gpt-oss.service

# Reboot to ensure NVIDIA drivers are loaded
reboot
"""

class GPTOSSAWSDeployer:
    """Deploy GPT OSS on AWS EC2."""
    
    def __init__(self, region: str = AWS_REGION):
        self.region = region
        self.ec2_client = boto3.client('ec2', region_name=region)
        self.ec2_resource = boto3.resource('ec2', region_name=region)
    
    def create_security_group(self) -> str:
        """Create security group for GPT OSS server."""
        try:
            # Check if security group already exists
            response = self.ec2_client.describe_security_groups(
                Filters=[{'Name': 'group-name', 'Values': [SECURITY_GROUP_NAME]}]
            )
            
            if response['SecurityGroups']:
                sg_id = response['SecurityGroups'][0]['GroupId']
                print(f"✅ Security group {SECURITY_GROUP_NAME} already exists: {sg_id}")
                return sg_id
            
            # Create new security group
            response = self.ec2_client.create_security_group(
                GroupName=SECURITY_GROUP_NAME,
                Description='Security group for GPT OSS server'
            )
            
            sg_id = response['GroupId']
            
            # Add rules
            self.ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 8000,
                        'ToPort': 8000,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'vLLM API server'}]
                    }
                ]
            )
            
            print(f"✅ Created security group: {sg_id}")
            return sg_id
            
        except Exception as e:
            print(f"❌ Error creating security group: {str(e)}")
            raise
    
    def create_key_pair(self) -> None:
        """Create EC2 key pair if it doesn't exist."""
        try:
            # Check if key pair exists
            response = self.ec2_client.describe_key_pairs(
                Filters=[{'Name': 'key-name', 'Values': [KEY_PAIR_NAME]}]
            )
            
            if response['KeyPairs']:
                print(f"✅ Key pair {KEY_PAIR_NAME} already exists")
                return
            
            # Create new key pair
            response = self.ec2_client.create_key_pair(KeyName=KEY_PAIR_NAME)
            
            # Save private key
            with open(f"{KEY_PAIR_NAME}.pem", 'w') as f:
                f.write(response['KeyMaterial'])
            
            print(f"✅ Created key pair: {KEY_PAIR_NAME}")
            print(f"💾 Private key saved to: {KEY_PAIR_NAME}.pem")
            print(f"🔒 Run: chmod 400 {KEY_PAIR_NAME}.pem")
            
        except Exception as e:
            print(f"❌ Error creating key pair: {str(e)}")
            raise
    
    def launch_instance(self, security_group_id: str) -> str:
        """Launch EC2 instance with GPT OSS setup."""
        try:
            response = self.ec2_client.run_instances(
                ImageId=AMI_ID,
                MinCount=1,
                MaxCount=1,
                InstanceType=INSTANCE_TYPE,
                KeyName=KEY_PAIR_NAME,
                SecurityGroupIds=[security_group_id],
                UserData=USER_DATA_SCRIPT,
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': 'GPT-OSS-Weather-Demo'},
                            {'Key': 'Project', 'Value': 'GPT-OSS-Tool-Calling'}
                        ]
                    }
                ],
                BlockDeviceMappings=[
                    {
                        'DeviceName': '/dev/xvda',
                        'Ebs': {
                            'VolumeSize': 50,  # 50GB storage
                            'VolumeType': 'gp3',
                            'DeleteOnTermination': True
                        }
                    }
                ]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            print(f"🚀 Launched instance: {instance_id}")
            print(f"⏳ Instance is starting up...")
            
            return instance_id
            
        except Exception as e:
            print(f"❌ Error launching instance: {str(e)}")
            raise
    
    def wait_for_instance(self, instance_id: str) -> str:
        """Wait for instance to be running and get public IP."""
        try:
            print("⏳ Waiting for instance to be running...")
            
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            # Get instance details
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            
            public_ip = instance.get('PublicIpAddress')
            print(f"✅ Instance is running!")
            print(f"🌐 Public IP: {public_ip}")
            
            return public_ip
            
        except Exception as e:
            print(f"❌ Error waiting for instance: {str(e)}")
            raise
    
    def deploy(self) -> Dict[str, str]:
        """Deploy GPT OSS on AWS EC2."""
        print("🚀 Starting GPT OSS deployment on AWS EC2...")
        print(f"📍 Region: {self.region}")
        print(f"💻 Instance Type: {INSTANCE_TYPE}")
        print(f"🤖 Model: gpt-oss-20b")
        print()
        
        # Create resources
        self.create_key_pair()
        security_group_id = self.create_security_group()
        instance_id = self.launch_instance(security_group_id)
        public_ip = self.wait_for_instance(instance_id)
        
        print("\n" + "="*50)
        print("🎉 Deployment initiated successfully!")
        print("="*50)
        print(f"Instance ID: {instance_id}")
        print(f"Public IP: {public_ip}")
        print(f"SSH Command: ssh -i {KEY_PAIR_NAME}.pem ec2-user@{public_ip}")
        print(f"API Endpoint: http://{public_ip}:8000/v1/chat/completions")
        print()
        print("⏳ The instance is still setting up (10-15 minutes):")
        print("   1. Installing NVIDIA drivers")
        print("   2. Installing Python dependencies") 
        print("   3. Downloading GPT OSS model (~40GB)")
        print("   4. Starting vLLM server")
        print()
        print("🔍 Check setup progress:")
        print(f"   ssh -i {KEY_PAIR_NAME}.pem ec2-user@{public_ip}")
        print("   sudo journalctl -u gpt-oss.service -f")
        print()
        print("🧪 Test when ready:")
        print(f"   curl http://{public_ip}:8000/v1/models")
        
        return {
            "instance_id": instance_id,
            "public_ip": public_ip,
            "api_endpoint": f"http://{public_ip}:8000",
            "ssh_command": f"ssh -i {KEY_PAIR_NAME}.pem ec2-user@{public_ip}"
        }

def main():
    """Main deployment function."""
    print("🌤️ GPT OSS AWS EC2 Deployment")
    print("=" * 40)
    
    # Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Account: {identity['Account']}")
        print(f"✅ User/Role: {identity['Arn']}")
    except Exception as e:
        print(f"❌ AWS credentials not configured: {str(e)}")
        print("💡 Run: aws configure")
        return
    
    # Deploy
    deployer = GPTOSSAWSDeployer()
    deployment_info = deployer.deploy()
    
    # Save deployment info
    with open('aws_deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"💾 Deployment info saved to: aws_deployment_info.json")

if __name__ == "__main__":
    main()
