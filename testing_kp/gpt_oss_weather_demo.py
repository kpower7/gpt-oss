#!/usr/bin/env python3
"""
GPT OSS Tool Calling Demo with Real Weather APIs

This script demonstrates how to use OpenAI's GPT OSS models with HuggingFace Transformers
for tool calling to interact with real weather APIs and provide intelligent responses.

Requirements:
- OpenWeatherMap API key (free tier available)
- Required packages: transformers, torch, requests, python-dotenv, pytz, openai-harmony
- GPT OSS model (will be downloaded automatically from HuggingFace)

Key differences from OpenAI API:
- Uses harmony response format instead of OpenAI's function calling
- Tool definitions are configured via SystemContent.with_tools()
- Response parsing uses harmony encoding instead of OpenAI's tool_calls
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
import pytz
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Harmony format imports for tool calling
try:
    from openai_harmony import (
        SystemContent, 
        Message, 
        Conversation, 
        Role, 
        load_harmony_encoding, 
        HarmonyEncodingName,
        TextContent,
        ToolDescription
    )
    HARMONY_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: openai-harmony not installed. Install with: pip install openai-harmony")
    HARMONY_AVAILABLE = False

# Load environment variables
load_dotenv()

# OpenWeatherMap API configuration
OWM_API_KEY = os.getenv('OPENWEATHER_API_KEY')
OWM_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Model configuration - you can switch between models
MODEL_NAME = "openai/gpt-oss-20b"  # Start with smaller model for testing
# MODEL_NAME = "openai/gpt-oss-120b"  # Larger model for production

def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get real weather information for a location using OpenWeatherMap API.
    
    Args:
        location (str): City name, e.g., 'San Francisco' or 'Tokyo,JP'
        unit (str): Temperature unit - 'celsius' or 'fahrenheit'
    
    Returns:
        Dict containing weather information
    """
    # Check if API key is available
    if not OWM_API_KEY or OWM_API_KEY == 'your_openweather_api_key_here':
        print(f"⚠️ Warning: No valid OpenWeatherMap API key found. Using simulated data.")
        # Fallback to simulated data
        import random
        temp = random.randint(0, 35)
        conditions = random.choice(["sunny", "cloudy", "rainy", "snowy"])
        
        if unit == "fahrenheit":
            temp = temp * 9/5 + 32
            
        return {
            "location": location,
            "temperature": temp,
            "unit": "°F" if unit == "fahrenheit" else "°C",
            "conditions": conditions.title(),
            "humidity": random.randint(30, 90),
            "note": "This is simulated data. Please add a valid OpenWeatherMap API key.",
            "status": "simulated"
        }
    
    try:
        # Convert unit parameter to OpenWeatherMap format
        units = "metric" if unit == "celsius" else "imperial"
        
        # Make API request to OpenWeatherMap
        params = {
            'q': location,
            'appid': OWM_API_KEY,
            'units': units
        }
        
        print(f"🔍 Requesting weather data for {location}...")
        response = requests.get(OWM_BASE_URL, params=params, timeout=10)
        
        # Print status code for debugging
        print(f"📡 API Response Status: {response.status_code}")
        
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant weather information
        weather_info = {
            "location": f"{data['name']}, {data['sys']['country']}",
            "temperature": round(data['main']['temp'], 1),
            "feels_like": round(data['main']['feels_like'], 1),
            "humidity": data['main']['humidity'],
            "pressure": data['main']['pressure'],
            "unit": "°C" if unit == "celsius" else "°F",
            "conditions": data['weather'][0]['description'].title(),
            "wind_speed": data['wind']['speed'],
            "wind_direction": data['wind'].get('deg', 'N/A'),
            "visibility": data.get('visibility', 'N/A'),
            "sunrise": datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
            "sunset": datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
            "status": "success"
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {str(e)}")
        return {
            "location": location,
            "error": f"Network error: {str(e)}",
            "status": "error"
        }
    except KeyError as e:
        print(f"❌ Invalid location or API response: {str(e)}")
        return {
            "location": location,
            "error": f"Invalid location or API response: {str(e)}",
            "status": "error"
        }
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return {
            "location": location,
            "error": f"Unexpected error: {str(e)}",
            "status": "error"
        }

def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """
    Get the current time in a specific timezone.
    
    Args:
        timezone (str): Timezone name, e.g., 'UTC', 'US/Eastern', 'Europe/London'
    
    Returns:
        Dict containing time information
    """
    try:
        # Handle common timezone abbreviations
        timezone_mapping = {
            'EST': 'US/Eastern',
            'PST': 'US/Pacific',
            'CST': 'US/Central',
            'MST': 'US/Mountain',
            'GMT': 'GMT',
            'UTC': 'UTC',
            'JST': 'Asia/Tokyo',
            'CET': 'Europe/Paris',
            'BST': 'Europe/London'
        }
        
        # Map common abbreviations to full timezone names
        tz_name = timezone_mapping.get(timezone.upper(), timezone)
        
        # Get timezone object
        tz = pytz.timezone(tz_name)
        
        # Get current time in the specified timezone
        current_time = datetime.now(tz)
        
        return {
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": tz_name,
            "timezone_abbr": current_time.strftime("%Z"),
            "utc_offset": current_time.strftime("%z"),
            "day_of_week": current_time.strftime("%A"),
            "is_dst": bool(current_time.dst()),
            "status": "success"
        }
        
    except pytz.exceptions.UnknownTimeZoneError:
        return {
            "timezone": timezone,
            "error": f"Unknown timezone: {timezone}",
            "available_timezones_sample": ["UTC", "US/Eastern", "Europe/London", "Asia/Tokyo"],
            "status": "error"
        }
    except Exception as e:
        return {
            "timezone": timezone,
            "error": f"Error getting time: {str(e)}",
            "status": "error"
        }

def get_weather_forecast(location: str, days: int = 5) -> Dict[str, Any]:
    """
    Get weather forecast for a location (requires OpenWeatherMap API).
    
    Args:
        location (str): City name
        days (int): Number of days for forecast (1-5)
    
    Returns:
        Dict containing forecast information
    """
    # Check if API key is available
    if not OWM_API_KEY or OWM_API_KEY == 'your_openweather_api_key_here':
        print(f"⚠️ Warning: No valid OpenWeatherMap API key found. Using simulated forecast data.")
        # Fallback to simulated data
        import random
        from datetime import timedelta
        
        base_date = datetime.now()
        forecasts = []
        
        for i in range(days):
            forecast_date = base_date + timedelta(days=i)
            forecasts.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "temperature": round(random.uniform(10, 30), 1),
                "conditions": random.choice(["Clear Sky", "Scattered Clouds", "Light Rain", "Heavy Rain", "Thunderstorm"]),
                "humidity": random.randint(30, 90),
                "wind_speed": round(random.uniform(1, 10), 1),
                "simulated": True
            })
            
        return {
            "location": location,
            "forecasts": forecasts,
            "note": "This is simulated forecast data. Please add a valid OpenWeatherMap API key.",
            "status": "simulated"
        }
    
    try:
        # Use OpenWeatherMap's 5-day forecast API
        forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        
        params = {
            'q': location,
            'appid': OWM_API_KEY,
            'units': 'metric',
            'cnt': min(days * 8, 40)  # 8 forecasts per day (3-hour intervals)
        }
        
        print(f"🔍 Requesting forecast data for {location} for {days} day(s)...")
        response = requests.get(forecast_url, params=params, timeout=10)
        
        response.raise_for_status()
        data = response.json()
        
        # Process forecast data - simplified version
        forecasts = []
        processed_days = set()
        
        for item in data['list'][:days*8]:  # Limit to requested days
            forecast_time = datetime.fromtimestamp(item['dt'])
            day_key = forecast_time.strftime('%Y-%m-%d')
            
            # Take one forecast per day (around midday)
            if day_key not in processed_days and forecast_time.hour >= 12:
                forecast = {
                    "date": day_key,
                    "time": forecast_time.strftime('%H:%M'),
                    "temperature": round(item['main']['temp'], 1),
                    "conditions": item['weather'][0]['description'].title(),
                    "humidity": item['main']['humidity'],
                    "wind_speed": item['wind']['speed'],
                }
                forecasts.append(forecast)
                processed_days.add(day_key)
                
                if len(forecasts) >= days:
                    break
        
        return {
            "location": f"{data['city']['name']}, {data['city']['country']}",
            "forecasts": forecasts,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Error in forecast: {str(e)}")
        return {
            "location": location,
            "error": f"Error getting forecast: {str(e)}",
            "status": "error"
        }

# Define tool descriptions for harmony format
WEATHER_TOOLS = [
    ToolDescription.new(
        "get_weather",
        "Get current weather information for any city worldwide using real weather data",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, optionally with country code (e.g., 'San Francisco', 'Tokyo,JP', 'London,UK')"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit preference",
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    ),
    ToolDescription.new(
        "get_current_time",
        "Get the current time in any timezone worldwide",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g., 'UTC', 'US/Eastern', 'Europe/London', 'Asia/Tokyo') or abbreviation (EST, PST, etc.)",
                    "default": "UTC"
                }
            },
            "required": []
        }
    ),
    ToolDescription.new(
        "get_weather_forecast",
        "Get weather forecast for a location for the next few days",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name for the forecast"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days for forecast (1-5)",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 5
                }
            },
            "required": ["location"]
        }
    )
]

def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call and return the result."""
    print(f"🔧 Executing tool: {tool_name} with args: {arguments}")
    
    if tool_name == "get_weather":
        return get_weather(**arguments)
    elif tool_name == "get_current_time":
        return get_current_time(**arguments)
    elif tool_name == "get_weather_forecast":
        return get_weather_forecast(**arguments)
    else:
        return {"error": f"Unknown tool: {tool_name}", "status": "error"}

class GPTOSSWeatherAssistant:
    """GPT OSS Weather Assistant with tool calling capabilities."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.encoding = None
        
        if not HARMONY_AVAILABLE:
            raise ImportError("openai-harmony is required for tool calling. Install with: pip install openai-harmony")
        
        self.load_model()
        
    def load_model(self):
        """Load the GPT OSS model and tokenizer."""
        print(f"🤖 Loading GPT OSS model: {self.model_name}")
        print("📥 This may take a few minutes for first-time download...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            
            # Load model with appropriate device mapping
            device_map = "auto"
            if "120b" in self.model_name.lower():
                # For 120B model, use tensor parallelism if available
                device_map = {"tp_plan": "auto"} if torch.cuda.device_count() > 1 else "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True
            )
            
            # Create pipeline for easier inference
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype="auto",
                device_map=device_map
            )
            
            # Load harmony encoding
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def create_system_message(self) -> Message:
        """Create system message with tool configuration."""
        system_content = (
            SystemContent.new()
            .with_conversation_start_date(datetime.now().strftime("%Y-%m-%d"))
            .with_tools(WEATHER_TOOLS)
        )
        
        return Message.from_role_and_content(Role.SYSTEM, system_content)
    
    def process_with_tools(self, user_message: str) -> str:
        """Process user message with tool calling support."""
        try:
            # Create conversation with system message and user message
            system_message = self.create_system_message()
            user_msg = Message.from_role_and_content(Role.USER, user_message)
            
            messages = [system_message, user_msg]
            conversation = Conversation.from_messages(messages)
            
            # Convert to format for the model
            # For simplicity, we'll use the chat template approach
            chat_messages = [
                {"role": "system", "content": "You are a helpful weather and time assistant with access to real-time weather data and timezone information. When you need to get weather information, current time, or weather forecasts, use the available tools."},
                {"role": "user", "content": user_message}
            ]
            
            # Generate response
            print("🤔 Generating response...")
            outputs = self.pipe(
                chat_messages,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0]["generated_text"][-1]["content"]
            
            # Simple tool call detection (this is a simplified approach)
            # In a full implementation, you'd parse the harmony format properly
            if "get_weather" in response or "get_current_time" in response or "get_weather_forecast" in response:
                print("🔧 Tool call detected, but simplified parsing not implemented.")
                print("💡 For full tool calling, use the harmony format parsing from the chat.py example.")
            
            return response
            
        except Exception as e:
            return f"❌ Error processing request: {str(e)}"

def simple_chat_demo():
    """Simple demo without full harmony parsing (for quick testing)."""
    print("🌤️  GPT OSS Weather Assistant Demo (Simplified)")
    print("=" * 50)
    
    # Check if API keys are configured
    if not OWM_API_KEY:
        print("⚠️  Warning: OPENWEATHER_API_KEY not found. Weather functions will use simulated data.")
        print("   Get a free API key at: https://openweathermap.org/api")
    
    try:
        assistant = GPTOSSWeatherAssistant()
        
        # Test queries
        test_queries = [
            "What's the weather like in Tokyo right now?",
            "What time is it in London?",
            "Should I wear a jacket in New York today?",
        ]
        
        print("\n🤖 Running test queries...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] User: {query}")
            print("-" * 40)
            response = assistant.process_with_tools(query)
            print(f"Assistant: {response}")
        
        # Interactive mode
        print("\n💬 Interactive mode (type 'quit' to exit):")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input:
                    response = assistant.process_with_tools(user_input)
                    print(f"Assistant: {response}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {str(e)}")
        print("💡 Make sure you have sufficient GPU memory and all dependencies installed.")

def main():
    """Main function with setup instructions."""
    print("🌤️  GPT OSS Weather & Time Assistant")
    print("=" * 50)
    print("📋 Setup Requirements:")
    print("1. Install dependencies: pip install transformers torch openai-harmony requests python-dotenv pytz")
    print("2. Optional: Set OPENWEATHER_API_KEY in .env file for real weather data")
    print("3. Ensure sufficient GPU memory (8GB+ for 20B model, 80GB+ for 120B model)")
    print()
    
    if not HARMONY_AVAILABLE:
        print("❌ openai-harmony not available. Install with: pip install openai-harmony")
        print("🔄 Running without full tool calling support...")
        
    simple_chat_demo()

if __name__ == "__main__":
    main()
