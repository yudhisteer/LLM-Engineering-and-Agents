import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI
from utils import *
from website import Website

# Based on: https://github.com/ed-donner/llm_engineering/blob/main/week1/day1.ipynb

if __name__ == "__main__":
    WEBSITE_URL = "https://forbes.com/"
    # Create a website object
    website = Website(WEBSITE_URL)
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
    messages = configure_message(system_prompt, user_prompt)
    # print("Messages: ", messages)

    # Summarize the website
    response = summarize_website(system_prompt, user_prompt)
    # print("Response: ", response)

    # ------------------------------------------ Part 2 ------------------------------------------#
    # print("Contents: ", website.get_contents())
    # print("Links: ", website.links)

    # One-shot prompt - we provide one example of the output
    system_prompt_for_links = "You are provided with a list of links found on a webpage. \
    You are able to decide which of the links would be most relevant to include in a brochure about the company, \
    such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
    system_prompt_for_links += "You should respond in JSON as in this example:"
    system_prompt_for_links += """
    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page": "url": "https://another.full.url/careers"}
        ]
    }
    """
    # print(system_prompt_for_links)

    # Configure the user prompt for the links
    user_prompt_for_links = configure_user_prompt_for_links(website)
    # print("User Prompt for Links: ", user_prompt_for_links)

    # Configure the system prompt for the useful links
    useful_links_response = useful_links(system_prompt_for_links, user_prompt_for_links)
    # print("Useful Links: ", useful_links_response)

    # Extract all details
    all_details = extract_all_details(WEBSITE_URL, system_prompt_for_links, user_prompt_for_links)
    # print("All Details: ", all_details)

    system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
    and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
    Include details of company culture, customers and careers/jobs if you have the information."

    user_prompt = configure_user_prompt_brochure(
        WEBSITE_URL, "Forbes", system_prompt_for_links, user_prompt_for_links
    )
    # print("User Prompt for Brochure: ", user_prompt)

    brochure = create_brochure(system_prompt, user_prompt)
    print("Brochure: ", brochure)
