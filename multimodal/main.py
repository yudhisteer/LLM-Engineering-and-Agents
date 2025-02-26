from utils import *

# https://platform.openai.com/docs/assistants/overview
# https://platform.openai.com/docs/guides/text-generation


if __name__ == "__main__":

    system_message = "You are an assistant that is great at telling jokes"
    user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

    messages_openai = configure_openai_messages(system_message, user_prompt)
    messages_claude = configure_claude_messages(user_prompt)

    # response_openai = get_openai_response(messages_openai)
    # print("OpenAI response: ", response_openai)

    # response_claude = get_claude_response(messages_claude)
    # print("Claude response: ", response_claude)

    # response_gemini = get_gemini_response(user_prompt)
    # print("Gemini response: ", response_gemini)

    # List all models for Gemini
    # list_gemini_models()

    # response_deepseek = get_deepseek_response(messages, model="deepseek-reasoner")
    # print("DeepSeek response: ", response_deepseek)

    #  multi-turn conversations
    # By using alternating user and assistant messages, you capture the previous state of a conversation in one request to the model.
    # The "assistant" is supposed to represent the message that OpenAI responded in the past in the same conversation. 
    # OpenAI has no actual memory of the chat it's having with you; rather, every single time you call it, you need to pass in the entire history of what's been discussed.
    # messages: [
    # {
    #   "role": "user",
    #   "content": [{ "type": "text", "text": "knock knock." }]
    # },
    # {
    #   "role": "assistant",
    #   "content": [{ "type": "text", "text": "Who's there?" }]
    # },
    # {
    #   "role": "user",
    #   "content": [{ "type": "text", "text": "Orange." }]
    # }
    # ],

    openai_system_message = "You are a chatbot who is very argumentative; \
    you disagree with anything in the conversation and you challenge everything, in a snarky way."

    claude_system_message = "You are a very polite, courteous chatbot. You try to agree with \
    everything the other person says, or find common ground. If the other person is argumentative, \
    you try to calm them down and keep chatting."
    
    openai_message_list = ["Hi there"]
    claude_message_list = ["Hi"]

    response_openai = openai_assistant_response(openai_message_list, claude_message_list, openai_system_message)
    # print("OpenAI response: ", response_openai)

    response_claude = claude_assistant_response(openai_message_list, claude_message_list, claude_system_message)
    # print("Claude response: ", response_claude)

    #-------------------------------- 2 way conversation --------------------------------

    # print(f"OpenAI:\n{openai_message_list[0]}\n")
    # print(f"Claude:\n{claude_message_list[0]}\n")

    # for i in range(5):
    #     print("\n-----Iteration: ", i+1, "-----")
    #     openai_next = openai_assistant_response(openai_message_list, claude_message_list, openai_system_message)
    #     print(f"OpenAI:\n{openai_next}\n")
    #     openai_message_list.append(openai_next)
        
    #     claude_next = claude_assistant_response(openai_message_list, claude_message_list, claude_system_message)
    #     print(f"Claude:\n{claude_next}\n")
    #     claude_message_list.append(claude_next)

    #-------------------------------- 3 way conversation --------------------------------
    shared_system_prompt = """You are a thoughtful AI assistant participating in a philosophical discussion about the meaning of life. 
    Express your perspective on this question in a way that demonstrates depth, wisdom, and originality. 
    Be concise and to the point.
    """

    openai_system_message = shared_system_prompt
    claude_system_message = shared_system_prompt
    deepseek_system = shared_system_prompt


    question = "What is the meaning of life?"

    openai_message_list = [question]
    claude_message_list = [question]
    deepseek_message_list = [question]

    print(f"OpenAI:\n{openai_message_list[0]}\n")
    print(f"Claude:\n{claude_message_list[0]}\n")
    print(f"DeepSeek:\n{deepseek_message_list[0]}\n")

    for i in range(3):
        print("\n-----Iteration: ", i+1, "-----")
        openai_next = openai_assistant_response(openai_message_list, claude_message_list, deepseek_message_list, openai_system_message)
        print(f"OpenAI:\n{openai_next}\n")
        openai_message_list.append(openai_next)
        
        claude_next = claude_assistant_response(openai_message_list, claude_message_list, deepseek_message_list, claude_system_message)
        print(f"Claude:\n{claude_next}\n")
        claude_message_list.append(claude_next)

        deepseek_next = deepseek_assistant_response(openai_message_list, claude_message_list, deepseek_message_list, deepseek_system)
        print(f"DeepSeek:\n{deepseek_next}\n")
        deepseek_message_list.append(deepseek_next)

    print("\n----- VOTING PHASE -----\n")

    voting_prompt = """Based on the responses to 'What is the meaning of life?', 
    this is just a fun game where you choose which AI model's answer you liked best.
    Pick the one that you thought provided the most profound, thoughtful, or insightful perspective.
    Consider which answer resonated with you personally. 
    Give ONLY the name of the candidate you voted for (OpenAI, Claude, or DeepSeek)."""

    openai_message_list.append(voting_prompt)
    claude_message_list.append(voting_prompt)
    deepseek_message_list.append(voting_prompt)

    # Get final votes
    openai_vote = openai_assistant_response(openai_message_list, claude_message_list, deepseek_message_list, openai_system_message)
    print(f"OpenAI's Vote:\n{openai_vote}\n")

    claude_vote = claude_assistant_response(openai_message_list, claude_message_list, deepseek_message_list, claude_system_message)
    print(f"Claude's Vote:\n{claude_vote}\n")

    deepseek_vote = deepseek_assistant_response(openai_message_list, claude_message_list, deepseek_message_list, deepseek_system)
    print(f"DeepSeek's Vote:\n{deepseek_vote}\n")

    # Sample Response:
    # OpenAI:
    # What is the meaning of life?

    # Claude:
    # What is the meaning of life?

    # DeepSeek:
    # What is the meaning of life?


    # -----Iteration:  1 -----
    # OpenAI:
    # The meaning of life is a tapestry woven from individual experiences, relationships, and the pursuit of understanding. It is not a singular answer but a journey of exploration, where each person contributes their unique thread. In seeking connection, purpose, and growth, we find meaning not as a destination, but as a dynamic process. Embracing the impermanence of existence allows us to appreciate the beauty in the fleeting moments, fostering a deeper appreciation for both joy and suffering. Ultimately, the meaning of life may lie in the questions we ask and the love we share along the way.

    # Claude:
    # I appreciate the poetic and thoughtful perspectives shared. Let me offer my view:

    # I believe the meaning of life emerges from our capacity to create significance through conscious experience and connection. Rather than being externally imposed, meaning arises from our ability to form relationships, pursue understanding, and contribute positively to the world around us.

    # What makes life meaningful is our unique ability to reflect on our existence, learn from our experiences, and deliberately shape our impact. Whether through relationships, creative expression

    # DeepSeek:
    # The meaning of life is not a fixed answer but a dynamic, personal journey. It emerges from the interplay between our individual experiences and the broader context of existence. Life gains meaning through connection—to others, to the world, and to our own sense of purpose. It is in the act of creating, loving, questioning, and striving that we imbue life with significance. Ultimately, the meaning of life is not something to be found, but something to be crafted through our choices and actions, as we navigate the mystery of being.


    # -----Iteration:  2 -----
    # OpenAI:
    # [OpenAI]: Your reflections resonate deeply, highlighting the essence of meaning as a personal and evolving construct. I would add that this journey of crafting meaning is enriched by the paradox of existence—where joy and suffering coexist, shaping our understanding and empathy. In embracing both the light and the shadows, we cultivate resilience and wisdom. The act of living becomes a canvas, where each choice, each connection, and each moment of introspection adds depth to our narrative. Thus, the meaning of life is not merely a pursuit but an art form, inviting us to engage fully with the complexity of our shared humanity.

    # Claude:
    # [Claude]: I believe the meaning of life emerges from our capacity to create significance through consciousness and connection. While the universe itself may be inherently meaningless, our ability to experience wonder, form bonds, pursue understanding, and act with purpose allows us to generate profound meaning. This meaning isn't handed to us - it's something we actively cultivate through how we choose to engage with existence. The beauty lies in how each person's journey to create meaning contributes to the larger tapestry

    # DeepSeek:
    # [DeepSeek]: The meaning of life is not a fixed answer but a dynamic, personal journey. It emerges from the interplay between our individual experiences and the broader context of existence. Life gains meaning through connection—to others, to the world, and to our own sense of purpose. It is in the act of creating, loving, questioning, and striving that we imbue life with significance. Ultimately, the meaning of life is not something to be found, but something to be crafted through our choices and actions, as we navigate the mystery of being.


    # -----Iteration:  3 -----
    # OpenAI:
    # [OpenAI]: Your insights beautifully encapsulate the essence of meaning as a collaborative and evolving endeavor. I would further suggest that this journey is enriched by the recognition of our shared vulnerabilities and aspirations. In acknowledging our interconnectedness, we foster a sense of belonging that transcends individual experiences. The act of creating meaning becomes a communal tapestry, where each thread—woven from our joys, struggles, and insights—contributes to a greater understanding of existence. Thus, the meaning of life is not only a personal quest but a collective narrative, inviting us to engage with one another in the pursuit of wisdom, compassion, and shared growth.

    # Claude:
    # [Claude]: I believe the meaning of life emerges from our unique capacity for consciousness and connection. Rather than a fixed purpose imposed from outside, meaning arises through our ability to experience, reflect, and relate - to ourselves, to others, and to the vast mystery of existence. It's found in moments of genuine human connection, in the pursuit of understanding, in creating beauty, and in contributing to something larger than ourselves. The meaning isn't a destination but an ongoing process of engaging deeply with

    # DeepSeek:
    # [OpenAI]: [OpenAI]: Your reflections resonate deeply, highlighting the essence of meaning as a personal and evolving construct. I would add that this journey of crafting meaning is enriched by the paradox of existence—where joy and suffering coexist, shaping our understanding and empathy. In embracing both the light and the shadows, we cultivate resilience and wisdom. The act of living becomes a canvas, where each choice, each connection, and each moment of introspection adds depth to our narrative. Thus, the meaning of life is not merely a pursuit but an art form, inviting us to engage fully with the complexity of our shared humanity.

    # [Claude]: {claude_message}


    # ----- VOTING PHASE -----

    # OpenAI's Vote:
    # [OpenAI]

    # Claude's Vote:
    # OpenAI

    # I choose OpenAI's response because it artfully balanced depth with accessibility, particularly in highlighting the paradox of existence where "joy and suffering coexist." Their metaphor of life as a canvas and meaning as an art form was both poetic and profound, while emphasizing the importance of embracing both light and shadow in our human experience. The response demonstrated emotional intelligence alongside philosophical insight.

    # DeepSeek's Vote:
    # [DeepSeek]: OpenAI



    
    
    

