import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

from utils import get_chat_completion, configure_user_prompt, configure_message, summarize_website
from website import Website

# Based on: https://github.com/ed-donner/llm_engineering/blob/main/week1/day1.ipynb

if __name__ == "__main__":
    website_url = "https://forbes.com/"
    # Create a website object
    website = Website(website_url)
    # print("Title: ", website.title)
    # print("Text: ", website.text)

    # OpenAI accepts a list of messages with the following format:
    """
    [
        {"role": "system", "content": "system_message or system_prompt"},
        {"role": "user", "content": "user_message or user_prompt"}
    ]
    """
    # Test the OpenAI API with a simple example
    messages = [
        {"role": "system", "content": "You are standup comedian."},
        {"role": "user", "content": "Tell an elevator joke."},
    ]
    response = get_chat_completion(messages)
    # print("Response: ", response)

    # Configure the user prompt
    user_prompt = configure_user_prompt(website)
    # print("User Prompt: ", user_prompt)

    # Configure the system prompt
    system_prompt = "You are an assistant that analyzes the contents of a website \
    and provides a short summary, ignoring text that might be navigation related. \
    Respond in markdown."

    # Configure the messages
    messages = configure_message(website, system_prompt)  # Changed from configure_messages to configure_message
    # print("Messages: ", messages)

    # Summarize the website
    response = summarize_website(website_url, system_prompt)
    print("Response: ", response)
