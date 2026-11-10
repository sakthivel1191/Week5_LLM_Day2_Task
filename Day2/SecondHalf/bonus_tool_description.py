import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL = "openai/gpt-4o-mini"
vague_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get information using an ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Get information using an ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"}
                },
                "required": ["customer_id"]
            }
        }
    }
]
clear_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Use only for checking order shipping status "
                "and delivery information. "
                "Accepts an order ID such as ORD101."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": (
                "Use only for retrieving customer profile "
                "information, including customer name and email. "
                "Accepts a customer ID such as CUST101."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"}
                },
                "required": ["customer_id"]
            }
        }
    }
]

question = "What is the shipping status of order ORD101?"
def test_tool_description(tool_list, title):
    print(title)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        tools=tool_list,
        tool_choice="auto"
    )
    message = response.choices[0].message
    if message.tool_calls:
        for tool in message.tool_calls:
            print(
                "Selected tool:",
                tool.function.name
            )
            print(
                "Arguments:",
                tool.function.arguments
            )
    else:
        print("No tool selected:", message.content)
test_tool_description(
    vague_tools,
    "VAGUE TOOL DESCRIPTIONS"
)
test_tool_description(
    clear_tools,
    "CLEAR TOOL DESCRIPTIONS"
)
client.close()
