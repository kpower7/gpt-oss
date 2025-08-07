# GPT OSS with Ollama - Setup Guide

## 🚀 Quick Start

Since you already have Ollama set up with the 20B model, you're almost ready to go!

### Prerequisites ✅
- [x] Ollama installed
- [x] GPT OSS 20B model downloaded (`ollama pull gpt-oss:20b`)
- [x] Model available on your other laptop

### Required Python Packages
```bash
pip install openai requests python-dotenv pytz
```

### Optional: Weather API Key
1. Get free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Create `.env` file in this directory:
```
OPENWEATHER_API_KEY=your_api_key_here
```

## 🎯 Running the Demo

### 1. Start Ollama (if not running as service)
```bash
ollama serve
```

### 2. Verify Model is Available
```bash
ollama list
# Should show gpt-oss:20b in the list
```

### 3. Test Basic Model
```bash
ollama run gpt-oss:20b "Hello, can you introduce yourself?"
```

### 4. Run the Weather Tool Calling Demo
```bash
python gpt_oss_ollama_weather.py
```

## 🔧 How It Works

### OpenAI-Compatible API
- Ollama runs at `http://localhost:11434/v1`
- Uses standard OpenAI client library
- Same tool calling format as OpenAI API
- No complex harmony format parsing needed!

### Tool Calling Flow
1. User asks weather question
2. GPT OSS decides to call weather function
3. Script executes real API call
4. Results fed back to GPT OSS
5. GPT OSS provides natural language response

### Example Interaction
```
User: "What's the weather in Tokyo?"

🔧 Calling function: get_weather with args: {'location': 'Tokyo', 'unit': 'celsius'}
🔍 Requesting weather data for Tokyo...
📡 API Response Status: 200
