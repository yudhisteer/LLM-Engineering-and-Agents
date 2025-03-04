import gradio as gr
from utils import *



def shout(text: str) -> str:
    return text.upper()


def simple_interface(function: callable) -> gr.Interface:
    return gr.Interface(
        fn=function,
        inputs="text",
        outputs="text",
        allow_flagging="never",
    )

def simple_interface_with_textbox(function: callable) -> gr.Interface:

    return gr.Interface(
        fn=function,
        inputs=[gr.Textbox(label="Your message:", lines=6)],
        outputs=[gr.Textbox(label="Response:", lines=8)],
        allow_flagging="never",
    )

def simple_interface_stream(function: callable) -> gr.Interface:

    return gr.Interface(
        fn=function,
        inputs=[gr.Textbox(label="Your message:", lines=6)],
        outputs=[gr.Markdown(label="Response:")],
        allow_flagging="never",
    )

def get_openai_response(
    user_prompt: str,
    model: str = "gpt-4o-mini",
) -> str:
    try:
        system_message = "You are a helpful assistant"
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
      ],
        )
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error getting chat completion: {str(e)}")

def get_openai_stream(
    user_prompt: str,
    model: str = "gpt-4o-mini",
):
    try:
        system_message = "You are a helpful assistant"
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
      ],
      stream=True,
        )
        result = ""
        for chunk in response:
            result += chunk.choices[0].delta.content or ""
            yield result

    except Exception as e:
        raise Exception(f"Error getting chat completion: {str(e)}")

if __name__ == "__main__":

    # simple_interface(shout).launch(share=True)

    simple_interface_stream(get_openai_stream).launch(share=True)
