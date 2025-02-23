import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from website import Website
import json

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")
client = OpenAI(api_key=api_key)


def get_chat_completion(
    messages: list[dict], 
    model: str = "gpt-4o-mini", 
    temperature: float = 0
    ) -> str:
    """
    Get a chat completion from OpenAI.

    Args:
        prompt: The text prompt to send
        model: The model to use (default: gpt-4)
        temperature: Controls randomness (0-1, default: 0)

    Returns:
        The completion text
    """
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error getting chat completion: {str(e)}")


def configure_user_prompt(website: Website) -> str:
    user_prompt = f"You are looking at a website titled {website.title}"

    user_prompt += """\nThe contents of this website is as follows; \
                   please provide a short summary of this website in markdown. \
                   If it includes news or announcements, then summarize these too.\n\n"""
    user_prompt += website.text

    return user_prompt


def configure_message(website: Website, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": configure_user_prompt(website)},
    ]

def summarize_website(url: str, system_prompt: str) -> str:
    website = Website(url)
    response = get_chat_completion(configure_message(website, system_prompt))
    return response

def check_ollama_model_exists(model_name: str) -> bool:
    """Check if the specified model is already downloaded in Ollama."""
    try:
        response = requests.get(f"http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = [model['name'] for model in response.json()['models']]
            exists = model_name in available_models
            print(f"Model '{model_name}' {'exists' if exists else 'does not exist'} in Ollama")
            print("Available models: ", available_models)
    except requests.RequestException:
        print("Error: Cannot connect to Ollama server. Make sure Ollama is running.")
        return False

def get_ollama_response(model: str, messages: list[dict]) -> str:
        try:
            response = requests.post(
                url="http://localhost:11434/api/chat",
                headers={"Content-Type": "application/json"},
                json={"model": model, "messages": messages}
            )
            response.raise_for_status()

            full_content = ""
            for chunk in response.iter_lines():
                if chunk:
                    data = json.loads(chunk)
                    if 'message' in data and 'content' in data['message']:
                        full_content += data['message']['content']

            return full_content
        except requests.RequestException as e:
            return f"Request failed: {str(e)}"
        except json.JSONDecodeError:
            return "Error decoding JSON response."