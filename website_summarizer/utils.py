import os
from dotenv import load_dotenv
from openai import OpenAI
from website import Website


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

