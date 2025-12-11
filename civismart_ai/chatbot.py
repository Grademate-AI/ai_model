import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def get_chatbot_response(user_message: str, use_openai: bool = True) -> str:
    """
    Returns a chatbot response using OpenAI GPT.
    Params:
        user_message: str
        use_openai: bool - if False, returns a demo message
    """
    if not use_openai or not OPENAI_API_KEY:
        return "Chatbot unavailable: OpenAI API key not set or OpenAI disabled."

    try:
        # Updated for OpenAI Python v1+
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are CiviSmart AI assistant. Answer volunteer questions about civic reports, tasks, points, and urgency clearly and concisely."
                },
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        return f"Error: {str(e)}"
