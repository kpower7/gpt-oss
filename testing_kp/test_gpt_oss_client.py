#!/usr/bin/env python3
"""
GPT OSS Client Test Script

This script tests the deployed GPT OSS model with tool calling capabilities.
It can work with either AWS EC2 or Azure VM deployments running vLLM.

Usage:
    python test_gpt_oss_client.py --endpoint http://YOUR_VM_IP:8000
"""

import requests
import json
import argparse
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GPTOSSClient:
    """Client for testing GPT OSS deployed with vLLM."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def test_connection(self) -> bool:
        """Test if the server is responding."""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                print(f"✅ Server is responding")
                print(f"📋 Available models: {[m['id'] for m in models['data']]}")
                return True
            else:
                print(f"❌ Server responded with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send chat completion request."""
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def test_basic_chat(self) -> None:
        """Test basic chat functionality."""
        print("\n🧪 Testing basic chat...")
        
        messages = [
            {"role": "user", "content": "Hello! Can you introduce yourself?"}
        ]
        
        response = self.chat_completion(messages)
        
        if "error" in response:
            print(f"❌ Error: {response['error']}")
        else:
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"🤖 Assistant: {assistant_message}")
    
    def test_weather_questions(self) -> None:
        """Test weather-related questions (without actual tool calling for now)."""
        print("\n🌤️ Testing weather questions...")
        
        questions = [
            "What's the weather like in Tokyo?",
            "Should I wear a jacket in London today?",
            "What time is it in New York?",
            "Can you get me a 5-day forecast for Paris?"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful weather assistant. When users ask about weather, time, or forecasts, explain that you would normally use weather APIs to get real-time data, but for this demo you can provide general information."
                },
                {"role": "user", "content": question}
            ]
            
            response = self.chat_completion(messages, max_tokens=256)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
            else:
                assistant_message = response["choices"][0]["message"]["content"]
                print(f"🤖 Assistant: {assistant_message[:200]}...")
    
    def test_reasoning(self) -> None:
        """Test reasoning capabilities."""
        print("\n🧠 Testing reasoning...")
        
        messages = [
            {
                "role": "user", 
                "content": "I'm planning a trip to Japan in March. What should I consider about the weather, and what would be the best tools to check current conditions?"
            }
        ]
        
        response = self.chat_completion(messages, max_tokens=400)
        
        if "error" in response:
            print(f"❌ Error: {response['error']}")
        else:
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"🤖 Assistant: {assistant_message}")
    
    def interactive_chat(self) -> None:
        """Interactive chat session."""
        print("\n💬 Interactive chat mode (type 'quit' to exit)")
        print("-" * 50)
        
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant running on GPT OSS. You can discuss weather, time, and general topics. Be friendly and informative."
            }
        ]
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                conversation.append({"role": "user", "content": user_input})
                
                print("🤔 Thinking...")
                response = self.chat_completion(conversation, max_tokens=512)
                
                if "error" in response:
                    print(f"❌ Error: {response['error']}")
                else:
                    assistant_message = response["choices"][0]["message"]["content"]
                    print(f"🤖 Assistant: {assistant_message}")
                    conversation.append({"role": "assistant", "content": assistant_message})
                
                # Keep conversation manageable
                if len(conversation) > 10:
                    conversation = conversation[:1] + conversation[-8:]
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def load_deployment_info(platform: str) -> Dict[str, str]:
    """Load deployment info from JSON files."""
    filename = f"{platform}_deployment_info.json"
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        print(f"⚠️ No deployment info found for {platform}")
        return {}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test GPT OSS deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="API endpoint URL (e.g., http://1.2.3.4:8000)"
    )
    parser.add_argument(
        "--platform",
        choices=["aws", "azure"],
        help="Load endpoint from deployment info file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat session"
    )
    
    args = parser.parse_args()
    
    print("🧪 GPT OSS Client Test")
    print("=" * 30)
    
    # Determine endpoint
    endpoint = None
    
    if args.endpoint:
        endpoint = args.endpoint
    elif args.platform:
        deployment_info = load_deployment_info(args.platform)
        if deployment_info and "api_endpoint" in deployment_info:
            endpoint = deployment_info["api_endpoint"]
            print(f"📍 Using {args.platform.upper()} endpoint: {endpoint}")
        else:
            print(f"❌ No deployment info found for {args.platform}")
            return
    else:
        # Try to auto-detect from deployment files
        for platform in ["aws", "azure"]:
            deployment_info = load_deployment_info(platform)
            if deployment_info and "api_endpoint" in deployment_info:
                endpoint = deployment_info["api_endpoint"]
                print(f"📍 Auto-detected {platform.upper()} endpoint: {endpoint}")
                break
        
        if not endpoint:
            print("❌ No endpoint specified and no deployment info found")
            print("💡 Usage:")
            print("   python test_gpt_oss_client.py --endpoint http://YOUR_VM_IP:8000")
            print("   python test_gpt_oss_client.py --platform aws")
            print("   python test_gpt_oss_client.py --platform azure")
            return
    
    # Create client and test
    client = GPTOSSClient(endpoint)
    
    # Test connection
    if not client.test_connection():
        print("💡 Make sure your VM is running and the service has started")
        print("🔍 Check service status with:")
        print("   ssh user@your-vm-ip")
        print("   sudo journalctl -u gpt-oss.service -f")
        return
    
    # Run tests
    if args.interactive:
        client.interactive_chat()
    else:
        client.test_basic_chat()
        client.test_weather_questions()
        client.test_reasoning()
        
        print("\n🎉 All tests completed!")
        print("💬 Run with --interactive for chat mode")

if __name__ == "__main__":
    main()
