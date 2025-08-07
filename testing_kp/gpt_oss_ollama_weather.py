#!/usr/bin/env python3
"""
GPT OSS Tool Calling Demo with Ollama and Real Weather APIs

This script demonstrates how to use OpenAI's GPT OSS models via Ollama
for tool calling to interact with real weather APIs and provide intelligent responses.

Requirements:
- Ollama installed and running with gpt-oss:20b model
- OpenWeatherMap API key (free tier available)
- Required packages: openai, requests, python-dotenv, pytz

Setup:
1. Install Ollama: https://ollama.com/download
2. Download model: ollama pull gpt-oss:20b
3. Start Ollama: ollama serve (if not running as service)
4. Run this script!

Key advantages of Ollama approach:
- Uses familiar OpenAI client library
- No complex harmony format parsing needed
- Local deployment, no cloud costs
- OpenAI-compatible API at localhost:11434
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import pytz
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "gpt-oss:20b"

# OpenWeatherMap API configuration
OWM_API_KEY = os.getenv('OPENWEATHER_API_KEY')
OWM_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

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

# Define the function specifications for the LLM (same as original OpenAI format!)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for any city worldwide using real weather data",
            "parameters": {
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in any timezone worldwide",
            "parameters": {
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get weather forecast for a location for the next few days",
            "parameters": {
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
            "content": "You are a helpful weather and time assistant powered by GPT OSS running locally via Ollama. You have access to real-time weather data and timezone information. Follow these guidelines: 1. Provide detailed, accurate, and friendly responses based ONLY on the data returned by the API calls. 2. For weather forecasts, clearly indicate the specific time of day for each forecast. 3. When providing weather information, include relevant details like temperature, conditions, humidity, and wind when available. 4. Offer practical advice based on the weather conditions. 5. For time queries, be precise about timezone information. 6. If the data shows 'simulated' status, clearly inform the user that you're providing simulated data, not real weather information."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    try:
        # First, get the model's response with potential function calls
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Let the model decide when to use functions
            temperature=0.7
        )

        response_message = response.choices[0].message
        messages.append(response_message)  # Add response to conversation history

        # Check if the model wants to call a function
        if response_message.tool_calls:
            # Process each function call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"🔧 Calling function: {function_name} with args: {function_args}")

                # Execute the function
                function_response = None
                if function_name == "get_weather":
                    function_response = get_weather(**function_args)
                elif function_name == "get_current_time":
                    function_response = get_current_time(**function_args)
                elif function_name == "get_weather_forecast":
                    function_response = get_weather_forecast(**function_args)

                # Add function response to messages
                if function_response:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(function_response)
                    })

            # Get a new response from the model after function call
            second_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7
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
        # Test basic completion
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello! Just testing the connection."}],
            max_tokens=50
        )
        
        if response.choices[0].message.content:
            print("✅ Ollama connection successful!")
            print(f"🤖 Model: {MODEL_NAME}")
            return True
        else:
            print("❌ Ollama responded but no content received")
            return False
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {str(e)}")
        print("💡 Make sure Ollama is running:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Verify model: ollama list")
        print("   3. Pull model if needed: ollama pull gpt-oss:20b")
        return False

def main():
    """
    Main function to run the GPT OSS Ollama tool calling demo.
    """
    print("🌤️  GPT OSS Weather & Time Assistant (via Ollama)")
    print("=" * 55)
    
    # Initialize OpenAI client for Ollama
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"  # Ollama doesn't require a real API key
    )
    
    # Test Ollama connection
    if not test_ollama_connection(client):
        return
    
    # Check if weather API key is configured
    if not OWM_API_KEY:
        print("⚠️  Warning: OPENWEATHER_API_KEY not found. Weather functions will use simulated data.")
        print("   Get a free API key at: https://openweathermap.org/api")
    
    # Test queries demonstrating various capabilities
    test_queries = [
        "What's the weather like in Tokyo right now?",
        # "What time is it in London?",
        # "I'm planning a trip to Paris next week. What's the weather forecast?",
        # "Should I wear a jacket in New York today?",
        # "What's the time difference between San Francisco and Sydney?",
        # "Is it raining in Seattle?"
    ]
    
    print("\n🤖 Running test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/6] User: {query}")
        print("-" * 40)
        response = assistant_with_tools(query, client)
        print(f"Assistant: {response}")
        print()
    
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
                response = assistant_with_tools(user_input, client)
                print(f"Assistant: {response}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
