import requests
from utils import check_ollama_model_exists, get_ollama_response
import json



if __name__ == "__main__":

    # Use ollama model
    MODEL = "llama3.1"

    # check if the model exists
    check_ollama_model_exists("llama3.1:latest")

    # User prompt
    messages = [
        {"role": "user", "content": "Tell a funny joke about a chicken."}
    ]

    # Inference
    response = get_ollama_response(MODEL, messages)
    print(response)