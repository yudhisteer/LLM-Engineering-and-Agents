from utils import *

if __name__ == "__main__":

    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

    response = get_anthropic_response(system_message, user_prompt)
    print("Anthropic response: ", response)

    response = get_openai_response(system_message, user_prompt)
    print("OpenAI response: ", response)

    response = get_gemini_response(user_prompt)
    print("Gemini response: ", response)

    # List all models for Gemini
    get_gemini_list_models()

    response = get_deepseek_response(system_message, user_prompt, model="deepseek-reasoner")
    print("DeepSeek response: ", response)
