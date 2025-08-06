#!/usr/bin/env python3
"""
Azure VM Deployment Script for GPT OSS Weather Demo

This script helps you deploy the GPT OSS weather tool calling demo on Azure VM
with GPU support using vLLM for OpenAI-compatible API serving.

Requirements:
- Azure CLI installed and logged in (az login)
- azure-mgmt-compute, azure-mgmt-network, azure-identity installed
- Your Azure subscription with GPU quota enabled
"""

import json
import time
from typing import Dict, Any
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient

# Configuration
SUBSCRIPTION_ID = None  # Will be detected automatically
RESOURCE_GROUP_NAME = "gpt-oss-rg"
LOCATION = "East US"  # Change to your preferred region
VM_NAME = "gpt-oss-vm"
VM_SIZE = "Standard_NC6s_v3"  # V100 GPU, 16GB VRAM
ADMIN_USERNAME = "azureuser"
VNET_NAME = "gpt-oss-vnet"
SUBNET_NAME = "gpt-oss-subnet"
NSG_NAME = "gpt-oss-nsg"
PUBLIC_IP_NAME = "gpt-oss-public-ip"
NIC_NAME = "gpt-oss-nic"

# Cloud-init script for automatic setup
CLOUD_INIT_SCRIPT = """#cloud-config
package_update: true
package_upgrade: true

packages:
  - nvidia-driver-470
  - nvidia-dkms-470
  - nvidia-utils-470
  - python3-pip
  - git
  - curl
  - wget

runcmd:
  # Install CUDA
  - wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
  - dpkg -i cuda-keyring_1.0-1_all.deb
  - apt-get update
  - apt-get -y install cuda-toolkit-11-8
  
  # Install Python dependencies
  - pip3 install --upgrade pip
  - pip3 install vllm transformers torch openai-harmony requests python-dotenv pytz
  
  # Clone repository
  - cd /home/azureuser
  - git clone https://github.com/openai/gpt-oss.git
  - chown -R azureuser:azureuser gpt-oss
  
  # Create startup script
  - |
    cat > /home/azureuser/start_gpt_oss.sh << 'EOF'
    #!/bin/bash
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Start vLLM server with GPT OSS 20B
    python3 -m vllm.entrypoints.openai.api_server \
        --model openai/gpt-oss-20b \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 4096
    EOF
  
  - chmod +x /home/azureuser/start_gpt_oss.sh
  - chown azureuser:azureuser /home/azureuser/start_gpt_oss.sh
  
  # Create systemd service
  - |
    cat > /etc/systemd/system/gpt-oss.service << 'EOF'
    [Unit]
    Description=GPT OSS vLLM Server
    After=network.target
    
    [Service]
    Type=simple
    User=azureuser
    WorkingDirectory=/home/azureuser
    ExecStart=/home/azureuser/start_gpt_oss.sh
    Restart=always
    RestartSec=10
    Environment=CUDA_VISIBLE_DEVICES=0
    Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    [Install]
    WantedBy=multi-user.target
    EOF
  
  - systemctl enable gpt-oss.service
  - systemctl daemon-reload
  
  # Reboot to ensure drivers are loaded
  - reboot

final_message: "GPT OSS setup completed. The system will reboot and start the service automatically."
"""

class GPTOSSAzureDeployer:
    """Deploy GPT OSS on Azure VM."""
    
    def __init__(self, subscription_id: str = None):
        self.credential = DefaultAzureCredential()
        
        # Get subscription ID if not provided
        if not subscription_id:
            from azure.mgmt.subscription import SubscriptionClient
            sub_client = SubscriptionClient(self.credential)
            subscriptions = list(sub_client.subscriptions.list())
            if subscriptions:
                self.subscription_id = subscriptions[0].subscription_id
                print(f"✅ Using subscription: {self.subscription_id}")
            else:
                raise ValueError("No Azure subscriptions found")
        else:
            self.subscription_id = subscription_id
        
        # Initialize clients
        self.resource_client = ResourceManagementClient(self.credential, self.subscription_id)
        self.compute_client = ComputeManagementClient(self.credential, self.subscription_id)
        self.network_client = NetworkManagementClient(self.credential, self.subscription_id)
    
    def create_resource_group(self) -> None:
        """Create resource group."""
        try:
            print(f"📁 Creating resource group: {RESOURCE_GROUP_NAME}")
            
            self.resource_client.resource_groups.create_or_update(
                RESOURCE_GROUP_NAME,
                {"location": LOCATION}
            )
            
            print(f"✅ Resource group created: {RESOURCE_GROUP_NAME}")
            
        except Exception as e:
            print(f"❌ Error creating resource group: {str(e)}")
            raise
    
    def create_network_resources(self) -> Dict[str, str]:
        """Create networking resources."""
        try:
            print("🌐 Creating network resources...")
            
            # Create virtual network
            vnet_params = {
                "location": LOCATION,
                "address_space": {"address_prefixes": ["10.0.0.0/16"]}
            }
            
            vnet_operation = self.network_client.virtual_networks.begin_create_or_update(
                RESOURCE_GROUP_NAME, VNET_NAME, vnet_params
            )
            vnet_operation.wait()
            
            # Create subnet
            subnet_params = {"address_prefix": "10.0.0.0/24"}
            
            subnet_operation = self.network_client.subnets.begin_create_or_update(
                RESOURCE_GROUP_NAME, VNET_NAME, SUBNET_NAME, subnet_params
            )
            subnet_result = subnet_operation.result()
            
            # Create network security group
            nsg_params = {
                "location": LOCATION,
                "security_rules": [
                    {
                        "name": "SSH",
                        "protocol": "Tcp",
                        "source_port_range": "*",
                        "destination_port_range": "22",
                        "source_address_prefix": "*",
                        "destination_address_prefix": "*",
                        "access": "Allow",
                        "priority": 1000,
                        "direction": "Inbound"
                    },
                    {
                        "name": "vLLM-API",
                        "protocol": "Tcp",
                        "source_port_range": "*",
                        "destination_port_range": "8000",
                        "source_address_prefix": "*",
                        "destination_address_prefix": "*",
                        "access": "Allow",
                        "priority": 1001,
                        "direction": "Inbound"
                    }
                ]
            }
            
            nsg_operation = self.network_client.network_security_groups.begin_create_or_update(
                RESOURCE_GROUP_NAME, NSG_NAME, nsg_params
            )
            nsg_result = nsg_operation.result()
            
            # Create public IP
            public_ip_params = {
                "location": LOCATION,
                "public_ip_allocation_method": "Static",
                "sku": {"name": "Standard"}
            }
            
            public_ip_operation = self.network_client.public_ip_addresses.begin_create_or_update(
                RESOURCE_GROUP_NAME, PUBLIC_IP_NAME, public_ip_params
            )
            public_ip_result = public_ip_operation.result()
            
            # Create network interface
            nic_params = {
                "location": LOCATION,
                "ip_configurations": [
                    {
                        "name": "ipconfig1",
                        "subnet": {"id": subnet_result.id},
                        "public_ip_address": {"id": public_ip_result.id}
                    }
                ],
                "network_security_group": {"id": nsg_result.id}
            }
            
            nic_operation = self.network_client.network_interfaces.begin_create_or_update(
                RESOURCE_GROUP_NAME, NIC_NAME, nic_params
            )
            nic_result = nic_operation.result()
            
            print("✅ Network resources created")
            
            return {
                "nic_id": nic_result.id,
                "public_ip_id": public_ip_result.id
            }
            
        except Exception as e:
            print(f"❌ Error creating network resources: {str(e)}")
            raise
    
    def create_vm(self, nic_id: str) -> str:
        """Create virtual machine."""
        try:
            print(f"💻 Creating VM: {VM_NAME}")
            
            vm_params = {
                "location": LOCATION,
                "os_profile": {
                    "computer_name": VM_NAME,
                    "admin_username": ADMIN_USERNAME,
                    "disable_password_authentication": True,
                    "linux_configuration": {
                        "ssh": {
                            "public_keys": [
                                {
                                    "path": f"/home/{ADMIN_USERNAME}/.ssh/authorized_keys",
                                    "key_data": self.get_ssh_public_key()
                                }
                            ]
                        }
                    },
                    "custom_data": self.encode_cloud_init()
                },
                "hardware_profile": {"vm_size": VM_SIZE},
                "storage_profile": {
                    "image_reference": {
                        "publisher": "Canonical",
                        "offer": "0001-com-ubuntu-server-focal",
                        "sku": "20_04-lts-gen2",
                        "version": "latest"
                    },
                    "os_disk": {
                        "name": f"{VM_NAME}-os-disk",
                        "caching": "ReadWrite",
                        "create_option": "FromImage",
                        "disk_size_gb": 128
                    }
                },
                "network_profile": {
                    "network_interfaces": [{"id": nic_id}]
                }
            }
            
            vm_operation = self.compute_client.virtual_machines.begin_create_or_update(
                RESOURCE_GROUP_NAME, VM_NAME, vm_params
            )
            
            print("⏳ VM creation in progress (this may take 10-15 minutes)...")
            vm_result = vm_operation.result()
            
            print(f"✅ VM created: {VM_NAME}")
            return vm_result.id
            
        except Exception as e:
            print(f"❌ Error creating VM: {str(e)}")
            raise
    
    def get_ssh_public_key(self) -> str:
        """Get SSH public key for VM access."""
        import os
        import subprocess
        
        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
        
        if os.path.exists(ssh_key_path):
            with open(ssh_key_path, 'r') as f:
                return f.read().strip()
        else:
            print("🔑 No SSH key found, generating one...")
            subprocess.run([
                "ssh-keygen", "-t", "rsa", "-b", "2048", 
                "-f", os.path.expanduser("~/.ssh/id_rsa"), 
                "-N", ""
            ], check=True)
            
            with open(ssh_key_path, 'r') as f:
                return f.read().strip()
    
    def encode_cloud_init(self) -> str:
        """Encode cloud-init script for VM."""
        import base64
        return base64.b64encode(CLOUD_INIT_SCRIPT.encode()).decode()
    
    def get_public_ip(self) -> str:
        """Get the public IP address of the VM."""
        try:
            public_ip = self.network_client.public_ip_addresses.get(
                RESOURCE_GROUP_NAME, PUBLIC_IP_NAME
            )
            return public_ip.ip_address
            
        except Exception as e:
            print(f"❌ Error getting public IP: {str(e)}")
            raise
    
    def deploy(self) -> Dict[str, str]:
        """Deploy GPT OSS on Azure VM."""
        print("🚀 Starting GPT OSS deployment on Azure VM...")
        print(f"📍 Location: {LOCATION}")
        print(f"💻 VM Size: {VM_SIZE}")
        print(f"🤖 Model: gpt-oss-20b")
        print()
        
        # Create resources
        self.create_resource_group()
        network_info = self.create_network_resources()
        vm_id = self.create_vm(network_info["nic_id"])
        
        # Wait a bit for public IP to be assigned
        time.sleep(30)
        public_ip = self.get_public_ip()
        
        print("\n" + "="*50)
        print("🎉 Deployment completed successfully!")
        print("="*50)
        print(f"VM Name: {VM_NAME}")
        print(f"Public IP: {public_ip}")
        print(f"SSH Command: ssh {ADMIN_USERNAME}@{public_ip}")
        print(f"API Endpoint: http://{public_ip}:8000/v1/chat/completions")
        print()
        print("⏳ The VM is still setting up (15-20 minutes):")
        print("   1. Installing NVIDIA drivers")
        print("   2. Installing Python dependencies")
        print("   3. Downloading GPT OSS model (~40GB)")
        print("   4. Starting vLLM server")
        print()
        print("🔍 Check setup progress:")
        print(f"   ssh {ADMIN_USERNAME}@{public_ip}")
        print("   sudo journalctl -u gpt-oss.service -f")
        print()
        print("🧪 Test when ready:")
        print(f"   curl http://{public_ip}:8000/v1/models")
        
        return {
            "vm_id": vm_id,
            "public_ip": public_ip,
            "api_endpoint": f"http://{public_ip}:8000",
            "ssh_command": f"ssh {ADMIN_USERNAME}@{public_ip}",
            "resource_group": RESOURCE_GROUP_NAME
        }

def main():
    """Main deployment function."""
    print("🌤️ GPT OSS Azure VM Deployment")
    print("=" * 40)
    
    # Check Azure credentials
    try:
        from azure.mgmt.subscription import SubscriptionClient
        credential = DefaultAzureCredential()
        sub_client = SubscriptionClient(credential)
        subscriptions = list(sub_client.subscriptions.list())
        
        if subscriptions:
            print(f"✅ Azure Subscription: {subscriptions[0].display_name}")
            print(f"✅ Subscription ID: {subscriptions[0].subscription_id}")
        else:
            raise Exception("No subscriptions found")
            
    except Exception as e:
        print(f"❌ Azure credentials not configured: {str(e)}")
        print("💡 Run: az login")
        return
    
    # Deploy
    deployer = GPTOSSAzureDeployer()
    deployment_info = deployer.deploy()
    
    # Save deployment info
    with open('azure_deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"💾 Deployment info saved to: azure_deployment_info.json")

if __name__ == "__main__":
    main()
