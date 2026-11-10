import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OpenRouter API key not found!")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    max_retries=0
)
MODEL = "openai/gpt-4o-mini"
weather_data = {
    "Chennai": {
        "temperature": 31,
        "condition": "light rain"
    },
    "Bangalore": {
        "temperature": 25,
        "condition": "cloudy"
    },
    "Mumbai": {
        "temperature": 30,
        "condition": "sunny"
    }
}
def get_weather(city):
    if city in weather_data:
        return weather_data[city]
    return {
        "error": "Weather information not available"
    }
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get weather information for a city. "
                "Use this tool when the user asks about "
                "weather, temperature, or rain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city"
                    }
                },
                "required": ["city"],
                "additionalProperties": False
            }
        }
    }
]
messages = [
    {
        "role": "user",
        "content": "Is it raining in Chennai?"
    }
]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
assistant_message = response.choices[0].message
messages.append(assistant_message.model_dump(exclude_none=True))
if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(
            tool_call.function.arguments
        )
        print(
            f"Tool requested: {function_name}({json.dumps(arguments)})"
        )
        if function_name == "get_weather":
            result = get_weather(**arguments)
        else:
            result = {"error": "Unknown tool"}
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    print("\nFinal:")
    print(final_response.choices[0].message.content)
else:
    print("Model did not request a tool.")
    print(assistant_message.content)
