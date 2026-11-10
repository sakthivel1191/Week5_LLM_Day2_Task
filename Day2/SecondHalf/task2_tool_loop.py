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
orders = {
    "ORD101": {
        "status": "Shipped",
        "delivery": "Tomorrow"
    },
    "ORD102": {
        "status": "Processing",
        "delivery": "Within 3 days"
    },
    "ORD103": {
        "status": "Delivered",
        "delivery": "Yesterday"
    }
}
customers = {
    "CUST101": {
        "name": "Arun",
        "email": "arun@example.com"
    },
    "CUST102": {
        "name": "Priya",
        "email": "priya@example.com"
    },
    "CUST103": {
        "name": "Kumar",
        "email": "kumar@example.com"
    }
}
def get_order_status(order_id):
    """Get the status and delivery details of an order."""
    if order_id in orders:
        return orders[order_id]
    return {
        "error": "Order not found"
    }
def get_customer(customer_id):
    """Get customer name and email using customer ID."""
    if customer_id in customers:
        return customers[customer_id]
    return {
        "error": "Customer not found"
    }
def calculate_discount(amount, percent):
    """Calculate discount amount and final price."""
    discount = amount * percent / 100
    final_price = amount - discount
    return {
        "original_amount": amount,
        "discount_percent": percent,
        "discount_amount": round(discount, 2),
        "final_price": round(final_price, 2)
    }
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Use this tool when the user asks about "
                "an order's shipping status, delivery status, "
                "or expected delivery date. "
                "Returns the order status and delivery estimate. "
                "Requires an order ID such as ORD101."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Unique order ID, e.g. ORD101"
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": (
                "Use this tool when the user asks for "
                "customer profile information, customer name, "
                "or customer email address. "
                "Returns customer name and email. "
                "Requires a customer ID such as CUST101."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Unique customer ID, e.g. CUST101"
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_discount",
            "description": (
                "Use this tool when the user asks to calculate "
                "a percentage discount, discount amount, "
                "or final price after discount. "
                "Returns the discount amount and final price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Original price or amount"
                    },
                    "percent": {
                        "type": "number",
                        "description": "Discount percentage"
                    }
                },
                "required": ["amount", "percent"],
                "additionalProperties": False
            }
        }
    }
]
available_functions = {
    "get_order_status": get_order_status,
    "get_customer": get_customer,
    "calculate_discount": calculate_discount
}
def run_tool_loop(question):
    print("\nUSER QUESTION:", question)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support assistant. "
                "Use the available tools when required. "
                "Never invent order or customer information. "
                "If information is missing, ask the user."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    for round_number in range(1, 6):
        print(f"\nLLM Round: {round_number}")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        assistant_message = response.choices[0].message
        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )
        if not assistant_message.tool_calls:
            print("\nFINAL ANSWER:")
            print(assistant_message.content)
            break
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(
                tool_call.function.arguments
            )
            print("\nTool requested:", function_name)
            print("Arguments:", arguments)
            function_to_call = available_functions.get(function_name)
            if function_to_call:
                try:
                    result = function_to_call(**arguments)
                except (TypeError, ValueError) as error:
                    result = {
                        "error": str(error)
                    }
            else:
                result = {
                    "error": "Unknown tool"
                }
            print("Tool result:", result)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    else:
        print("\nMaximum 5 rounds reached.")
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages + [
                {
                    "role": "system",
                    "content": (
                        "Give your best final answer using the "
                        "information already available. "
                        "Do not request additional tools."
                    )
                }
            ],
            tool_choice="none"
        )
        print("\nFINAL ANSWER:")
        print(final_response.choices[0].message.content)
    print("\nFULL MESSAGE HISTORY:")
    print(
        json.dumps(
            messages,
            indent=4,
            default=str
        )
    )
    return messages
questions = [
    "What is the status of order ORD101?",
    "Show me the customer details for CUST102.",
    "Calculate a 10 percent discount on 5000.",
    "What is the status of order ORD103?",
    "Explain what an LLM is in simple words."
]
if __name__ == "__main__":
    try:
        for question in questions:
            run_tool_loop(question)
    finally:
        client.close()
        