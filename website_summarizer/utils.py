import json
import os

import requests
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
    temperature: float = 0,
    response_format: dict = None,
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
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
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


def configure_user_prompt_for_links(website: Website) -> str:
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
                    Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


def configure_user_prompt_brochure(
    url: str,
    company_name: str,
    system_prompt_for_links: str,
    user_prompt_for_links: str,
) -> str:
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += extract_all_details(url, 
                                       system_prompt_for_links, 
                                       user_prompt_for_links
                                       )
    user_prompt = user_prompt[:5_000]  # Truncate if more than 5,000 characters
    return user_prompt


def configure_message(system_prompt: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def summarize_website(system_prompt: str, user_prompt: str) -> str:
    response = get_chat_completion(
        messages=configure_message(system_prompt, user_prompt)
    )
    return response


def useful_links(system_prompt: str, user_prompt: str) -> str:
    response = get_chat_completion(
        messages=configure_message(system_prompt, user_prompt),
        response_format={"type": "json_object"},
    )
    return response


def extract_all_details(url: str, system_prompt: str, user_prompt: str) -> str:
    try:
        # Get main page content
        result = "Landing page:\n"
        main_page = Website(url)
        result += main_page.get_contents()

        # Get and process relevant links
        links_response = useful_links(system_prompt, user_prompt)
        links_data = (
            json.loads(links_response)
            if isinstance(links_response, str)
            else links_response
        )

        if not links_data.get("links"):
            print("No relevant links found")
            return result

        # Process each relevant link
        for link in links_data["links"]:
            try:
                # print(f"Processing link: {link['url']}")
                result += f"\n\n{link.get('type', 'Related Page')}:\n"
                linked_page = Website(link["url"])
                result += linked_page.get_contents()
            except Exception as e:
                print(f"Error processing link {link['url']}: {str(e)}")
                continue
        return result

    except Exception as e:
        raise Exception(f"Error extracting website details: {str(e)}")


def create_brochure(system_prompt: str, user_prompt: str) -> str:
    response = get_chat_completion(
        messages=configure_message(system_prompt, user_prompt)
    )
    return response


def check_ollama_model_exists(model_name: str) -> bool:
    """
    Check if the specified model is already downloaded in Ollama.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """

    try:
        response = requests.get(f"http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json()["models"]]
            exists = model_name in available_models
            print(
                f"Model '{model_name}' {'exists' if exists else 'does not exist'} in Ollama"
            )
            print("Available models: ", available_models)
    except requests.RequestException:
        print("Error: Cannot connect to Ollama server. Make sure Ollama is running.")
        return False


def get_ollama_response(model: str, messages: list[dict]) -> str:
    """
    Get the response from the Ollama model.

    Parameters:
    model (str): The name of the model to use for generating the response.
    messages (list[dict]): A list of messages to send to the model, where each message is a dictionary
                           containing a 'role' (either 'user' or 'system') and 'content' (the message text).
                           Ex: [{"role": "user", "content": "Hello, how are you?"}]

    Returns:
    str: The generated response from the model, or an error message if the request fails.
    """

    try:
        response = requests.post(
            url="http://localhost:11434/api/chat",
            headers={"Content-Type": "application/json"},
            json={"model": model, "messages": messages},
        )
        response.raise_for_status()

        full_content = ""
        for chunk in response.iter_lines():
            if chunk:
                data = json.loads(chunk)
                if "message" in data and "content" in data["message"]:
                    full_content += data["message"]["content"]

        return full_content
    except requests.RequestException as e:
        return f"Request failed: {str(e)}"
    except json.JSONDecodeError:
        return "Error decoding JSON response."
