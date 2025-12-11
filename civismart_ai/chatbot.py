import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def get_chatbot_response(user_message: str, use_openai: bool = True) -> str:
    """
    Returns a chatbot response using OpenAI GPT.
    Falls back to a demo message if API key is missing or call fails.
    """
    if not use_openai or not OPENAI_API_KEY:
        return f"[Demo Chatbot] You asked: {user_message}. Response: Please follow the civic task instructions."

    try:
        # OpenAI v1+ API call
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
        # Fallback to demo response if API call fails (e.g., quota exceeded)
        return f"[Demo Chatbot] You asked: {user_message}. Response: Please follow the civic task instructions. (OpenAI error: {str(e)})"
