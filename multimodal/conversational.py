import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import logging


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"

def get_openai_stream(
    messages: dict,
    model: str = "gpt-4o-mini",
):
    try:
        system_message = "You are a helpful assistant"
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        result = ""
        for chunk in response:
            result += chunk.choices[0].delta.content or ""
            yield result

    except Exception as e:
        raise Exception(f"Error getting chat completion: {str(e)}")



def chat(message: str, history: list[dict]):

    """
    This function is used to chat with the assistant.
    message is the user's message.
    history is a list of dictionaries, each dictionary contains a role and content key.
    Example:
    [
    {"role": "system", "content": "system message here"},
    {"role": "user", "content": "first user prompt here"},
    {"role": "assistant", "content": "the assistant's response"},
    {"role": "user", "content": "the new user prompt"},
    ]   
    """
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)

    return get_openai_stream(messages)
if __name__ == "__main__":
    system_message = "You are a helpful assistant"
