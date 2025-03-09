import os
import glob
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"



def get_relevant_context(message: str, context: dict) -> list[str]:
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context  

def add_context(message: str, context: dict) -> str:
    relevant_context = get_relevant_context(message, context)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message


def chat(message: str, history: list[dict], context: dict):
    system_message = "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."
    messages = [{"role": "system", "content": system_message}] + history
    message = add_context(message, context)
    messages.append({"role": "user", "content": message})

    stream = openai_client.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

if __name__ == "__main__":

    context = {}

    # --------------------------------------------------------------
    # Step 1: Get employee data
    # --------------------------------------------------------------

    employees = glob.glob("rag/data/employees/*")
    print("Employee files found:", employees)

    for employee in employees:
        name = employee.split(' ')[-1][:-3]
        doc = ""
        with open(employee, "r", encoding="utf-8") as f:
            doc = f.read()
        context[name]=doc

    logger.info(f"Found {len(context)} employees")
    print("Context: ", context)
    print("Lancaster: ", context["Lancaster"])
    print("context.keys(): ", context.keys()) # Print all employee names in employees directory


    # --------------------------------------------------------------
    # Step 2: Get product data
    # --------------------------------------------------------------

    products = glob.glob("rag/data/products/*")

    for product in products:
        name = product.split(os.sep)[-1][:-3]
        doc = ""
        with open(product, "r", encoding="utf-8") as f:
            doc = f.read()
        context[name]=doc

    print("context.keys(): ", context.keys())


    # --------------------------------------------------------------
    # Step 3: DIY RAG
    # --------------------------------------------------------------

    result = get_relevant_context("Who is lancaster?", context)
    print("Result: ", result)

    result = get_relevant_context("Who is Avery and what is carllm?", context)
    print("Result: ", result)

    result = add_context("Who is Alex Lancaster?", context)
    print("Result: ", result)

    view = gr.ChatInterface(chat, type="messages").launch()


