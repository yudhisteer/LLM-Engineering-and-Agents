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


openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = Anthropic(api_key=anthropic_api_key)
genai.configure(api_key=google_api_key)
deepseek_via_openai_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

def get_gemini_list_models() -> None:
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


def get_openai_response(
    system_message: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    response_format: dict = None,
) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format=response_format,
        )
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error getting chat completion: {str(e)}")


def get_anthropic_response(
    system_message: str,
    user_prompt: str,
    model: str = "claude-3-5-sonnet-latest",
    temperature: float = 0.7,
    max_tokens: int = 100,
    ) -> str:
    message = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_message,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    return message.content[0].text


def get_deepseek_response(system_message: str, user_prompt: str, model: str = "deepseek-chat", temperature: float = 0.0) -> str:
    response = deepseek_via_openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
            ],
        temperature=temperature,
        )

    return response.choices[0].message.content


