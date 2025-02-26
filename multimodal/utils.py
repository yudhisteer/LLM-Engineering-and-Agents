import os

import google.generativeai as genai
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")
if not anthropic_api_key:
    raise ValueError("Anthropic API key not found in environment variables")
if not google_api_key:
    raise ValueError("Google API key not found in environment variables")
if not deepseek_api_key:
    raise ValueError("DeepSeek API key not found in environment variables")

# Initialize clients
openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = Anthropic(api_key=anthropic_api_key)
genai.configure(api_key=google_api_key)
deepseek_via_openai_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")



def list_gemini_models() -> None:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)


def get_gemini_response(user_prompt: str, model: str = "gemini-2.0-flash-exp") -> str:
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error getting Gemini response: {str(e)}")


def configure_openai_messages(system_message: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

def configure_claude_messages(user_prompt: str) -> list[dict]:
    return [
        {"role": "user", "content": user_prompt},
    ]

def get_openai_response(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    response_format: dict = None,
) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error getting chat completion: {str(e)}")


def get_claude_response(
    system_message: str,
    messages: list[dict],
    model: str = "claude-3-5-sonnet-latest",
    temperature: float = 0.7,
    max_tokens: int = 100,
    ) -> str:
    try:
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=messages,
            )
        return message.content[0].text

    except Exception as e:
        raise Exception(f"Error getting anthropic response: {str(e)}")


def get_deepseek_response(messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.0) -> str:
    try:
        response = deepseek_via_openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            )
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error getting deepseek response: {str(e)}")
    


def openai_assistant_response(openai_messages: list[str], claude_messages: list[str], 
                             deepseek_messages: list[str] = None, openai_system: str = "") -> str:
    
    messages = [{"role": "system", "content": openai_system}]
    
    # Check if deepseek_messages is provided
    if deepseek_messages is None:
        # Only use openai and claude messages
        for openai_message, claude_message in zip(openai_messages, claude_messages):
            messages.append({"role": "assistant", "content": openai_message})
            messages.append({"role": "user", "content": claude_message})
    else:
        # Use all three message lists
        for openai_message, claude_message, deepseek_message in zip(openai_messages, claude_messages, deepseek_messages):
            messages.append({"role": "assistant", "content": f"[OpenAI]: {openai_message}"})
            messages.append({"role": "user", "content": f"[Claude]: {claude_message}"})
            messages.append({"role": "user", "content": f"[DeepSeek]: {deepseek_message}"})
    
    response_openai = get_openai_response(messages)
    return response_openai


def claude_assistant_response(openai_messages: list[str], 
                              claude_messages: list[str], 
                             deepseek_messages: list[str] = None, 
                             claude_system: str = "") -> str:
    messages = []
    
    # Check if deepseek_messages is provided
    if deepseek_messages is None:
        # Only use openai and claude messages
        for openai_message, claude_message in zip(openai_messages, claude_messages):
            messages.append({"role": "user", "content": openai_message})
            messages.append({"role": "assistant", "content": claude_message})
    else:
        # Use all three message lists
        for openai_message, claude_message, deepseek_message in zip(openai_messages, claude_messages, deepseek_messages):
            messages.append({"role": "user", "content": f"[OpenAI]: {openai_message}"})
            messages.append({"role": "assistant", "content": "[Claude]: {claude_message}"})
            messages.append({"role": "user", "content": f"[DeepSeek]: {deepseek_message}"})
    
    # Add the final openai message
    messages.append({"role": "user", "content": openai_messages[-1]})
    
    response_claude = get_claude_response(claude_system, messages)
    return response_claude

def deepseek_assistant_response(openai_messages: list[str], 
                                claude_messages: list[str], 
                                deepseek_messages: list[str] = None, 
                                deepseek_system: str = "") -> str:
    
    messages = [{"role": "system", "content": deepseek_system}]

    for openai_message, claude_message, deepseek_message in zip(openai_messages, claude_messages, deepseek_messages):
        messages.append({"role": "user", "content": f"[OpenAI]: {openai_message}"})
        messages.append({"role": "user", "content": "[Claude]: {claude_message}"})
        messages.append({"role": "assistant", "content": f"[DeepSeek]: {deepseek_message}"})
    
    response_deepseek = get_deepseek_response(messages)
    return response_deepseek
