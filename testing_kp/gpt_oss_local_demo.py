#!/usr/bin/env python3
"""
GPT-OSS Local Demo with Ollama

A simple demonstration of GPT-OSS running locally via Ollama.
This script provides:
1. General query answering
2. Email drafting
3. Opening the user's default email client

No external APIs required - runs entirely locally!

Requirements:
- Ollama installed and running with gpt-oss:20b model
- Required packages: openai

Setup:
1. Install Ollama: https://ollama.com/download
2. Download model: ollama pull gpt-oss:20b
3. Start Ollama: ollama serve (if not running as service)
4. Run this script!
"""

import json
import subprocess
import sys
import platform
import urllib.parse
from typing import Dict, Any
from openai import OpenAI

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "gpt-oss:20b"

def draft_email(recipient: str, subject: str, content: str, cc: str = None) -> Dict[str, Any]:
    """
    Draft an email and prepare it for sending via the user's default email client.
    
    Args:
        recipient (str): Email address of the recipient
        subject (str): Email subject line
        content (str): Email body content
        cc (str, optional): CC email address
    
    Returns:
        Dict containing email information and status
    """
    try:
        # Clean and format the email content
        formatted_content = content.strip()
        
        # Create email data structure
        email_data = {
            "recipient": recipient,
            "subject": subject,
            "content": formatted_content,
            "cc": cc,
            "status": "drafted"
        }
        
        print(f"📧 Email drafted successfully!")
        print(f"   To: {recipient}")
        if cc:
            print(f"   CC: {cc}")
        print(f"   Subject: {subject}")
        print(f"   Content preview: {formatted_content[:100]}{'...' if len(formatted_content) > 100 else ''}")
        
        return email_data
        
    except Exception as e:
        return {
            "error": f"Error drafting email: {str(e)}",
            "status": "error"
        }

def open_email_client(recipient: str, subject: str, content: str, cc: str = None) -> Dict[str, Any]:
    """
    Open the user's default email client with a pre-filled email.
    
    Args:
        recipient (str): Email address of the recipient
        subject (str): Email subject line
        content (str): Email body content
        cc (str, optional): CC email address
    
    Returns:
        Dict containing operation status
    """
    try:
        # URL encode the email components
        encoded_subject = urllib.parse.quote(subject)
        encoded_content = urllib.parse.quote(content)
        
        # Build mailto URL
        mailto_url = f"mailto:{recipient}?subject={encoded_subject}&body={encoded_content}"
        
        if cc:
            encoded_cc = urllib.parse.quote(cc)
            mailto_url += f"&cc={encoded_cc}"
        
        # Detect operating system and open email client
        system = platform.system().lower()
        
        if system == "windows":
            subprocess.run(["start", mailto_url], shell=True, check=True)
        elif system == "darwin":  # macOS
            subprocess.run(["open", mailto_url], check=True)
        elif system == "linux":
            subprocess.run(["xdg-open", mailto_url], check=True)
        else:
            return {
                "error": f"Unsupported operating system: {system}",
                "status": "error"
            }
        
        print(f"✅ Email client opened successfully!")
        print(f"   The email should now be open in your default email application.")
        
        return {
            "recipient": recipient,
            "subject": subject,
            "status": "email_client_opened"
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "error": f"Failed to open email client: {str(e)}",
            "status": "error"
        }
    except Exception as e:
        return {
            "error": f"Unexpected error opening email client: {str(e)}",
            "status": "error"
        }

def get_system_info() -> Dict[str, Any]:
    """
    Get basic system information for demonstration purposes.
    
    Returns:
        Dict containing system information
    """
    try:
        system_info = {
            "operating_system": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
            "status": "success"
        }
        
        return system_info
        
    except Exception as e:
        return {
            "error": f"Error getting system info: {str(e)}",
            "status": "error"
        }

# Define the function specifications for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": "Draft an email with specified recipient, subject, and content",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Email address of the recipient"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject line for the email"
                    },
                    "content": {
                        "type": "string",
                        "description": "Main body content of the email"
                    },
                    "cc": {
                        "type": "string",
                        "description": "Optional CC email address"
                    }
                },
                "required": ["recipient", "subject", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_email_client",
            "description": "Open the user's default email client with a pre-filled email ready to send",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Email address of the recipient"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject line for the email"
                    },
                    "content": {
                        "type": "string",
                        "description": "Main body content of the email"
                    },
                    "cc": {
                        "type": "string",
                        "description": "Optional CC email address"
                    }
                },
                "required": ["recipient", "subject", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get basic information about the current system (OS, Python version, etc.)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def assistant_with_tools(user_message: str, client: OpenAI) -> str:
    """
    Process user messages and handle function calls using GPT OSS via Ollama.
    
    Args:
        user_message (str): The user's input message
        client (OpenAI): OpenAI client configured for Ollama
    
    Returns:
        str: The assistant's response
    """
    messages = [
        {
            "role": "system",
            "content": """You are a helpful local AI assistant powered by GPT-OSS running via Ollama. You can:

1. Answer general questions and have conversations
2. Draft emails for users with proper formatting
3. Open the user's email client with pre-filled emails ready to send
4. Provide system information

Guidelines:
- Be friendly, helpful, and conversational
- When drafting emails, make them professional and well-formatted
- Always ask for clarification if email details are missing (recipient, subject, etc.)
- Offer to open the email client after drafting emails
- Provide detailed and informative responses
- Remember that you're running entirely locally - no internet required!"""
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    try:
        print("🧠 Thinking... (this may take a few minutes)")
        
        # First, get the model's response with potential function calls
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            timeout=300  # 5 minute timeout for thinking model
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        # Check if the model wants to call a function
        if response_message.tool_calls:
            # Process each function call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"🔧 Calling function: {function_name}")

                # Execute the function
                function_response = None
                if function_name == "draft_email":
                    function_response = draft_email(**function_args)
                elif function_name == "open_email_client":
                    function_response = open_email_client(**function_args)
                elif function_name == "get_system_info":
                    function_response = get_system_info(**function_args)

                # Add function response to messages
                if function_response:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(function_response)
                    })

            # Get a new response from the model after function call
            print("🧠 Processing function results... (thinking)")
            second_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                timeout=300  # 5 minute timeout for thinking model
            )

            return second_response.choices[0].message.content
        else:
            # If no function was called, return the initial response
            return response_message.content
            
    except Exception as e:
        return f"❌ Error: {str(e)}. Please check that Ollama is running and the model is available."

def test_ollama_connection(client: OpenAI) -> bool:
    """Test if Ollama is running and the model is available."""
    try:
        print("🔍 Testing Ollama connection...")
        print("🧠 Note: GPT-OSS is a thinking model - this may take a few minutes...")
        
        # Increase timeout for thinking model
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
            max_tokens=50,
            timeout=180  # 3 minute timeout for thinking model
        )
        
        # Check if we got a response object and it has choices
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content is not None and content.strip():  # Check for non-empty content
                print("✅ Ollama connection successful!")
                print(f"🤖 Model: {MODEL_NAME}")
                print(f"💬 Test response: {content.strip()[:50]}...")
                return True
            else:
                # Even if content is empty, the connection works - this might be normal
                print("✅ Ollama connection successful (empty response is OK)!")
                print(f"🤖 Model: {MODEL_NAME}")
                return True
        else:
            print("❌ No response received from Ollama")
            return False
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {str(e)}")
        print("💡 Make sure Ollama is running:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Verify model: ollama list")
        print("   3. Pull model if needed: ollama pull gpt-oss:20b")
        print("   4. Try running: ollama run gpt-oss:20b 'hello'")
        return False

def main():
    """
    Main function to run the GPT-OSS local demo.
    """
    print("🤖 GPT-OSS Local Demo (via Ollama)")
    print("=" * 40)
    print("Features:")
    print("• General Q&A and conversation")
    print("• Email drafting and sending")
    print("• System information")
    print("• Fully local operation!")
    print()
    
    # Initialize OpenAI client for Ollama
    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't require a real API key
            timeout=300.0  # 5 minute timeout for thinking model
        )
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {str(e)}")
        return
    
    # Test Ollama connection
    if not test_ollama_connection(client):
        return
    
    # Demo queries
    demo_queries = [
        "Hello! Can you tell me about yourself?",
        "What's my system information?",
        "Help me draft an email to john@example.com about scheduling a meeting next week"
    ]
    
    print("\n🎯 Running demo queries...\n")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"[{i}/{len(demo_queries)}] User: {query}")
        print("-" * 50)
        response = assistant_with_tools(query, client)
        print(f"Assistant: {response}")
        print()
    
    # Interactive mode
    print("\n💬 Interactive mode (type 'quit' to exit):")
    print("-" * 50)
    print("Try asking me to:")
    print("• Draft an email for you")
    print("• Answer questions about anything")
    print("• Get system information")
    print("• Have a general conversation")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Goodbye! Thanks for trying the GPT-OSS local demo!")
                break
            
            if user_input:
                print()
                response = assistant_with_tools(user_input, client)
                print(f"Assistant: {response}")
                print()
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Thanks for trying the GPT-OSS local demo!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
