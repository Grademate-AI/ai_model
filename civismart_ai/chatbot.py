import os
import openai
import google.generativeai as genai

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure both APIs if keys exist
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_chatbot_response(user_message: str, use_openai: bool = True) -> str:
    """
    Returns a chatbot response using:
        1. OpenAI GPT (primary)
        2. Gemini (fallback)
        3. Demo response (final fallback)

    Safe for civic assistant behavior.
    """

    # -----------------------------
    # 1. TRY OPENAI
    # -----------------------------
    if use_openai and OPENAI_API_KEY:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are CiviSmart AI assistant. Answer volunteer "
                            "questions about civic reports, tasks, points, and urgency clearly and concisely."
                        )
                    },
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content

        except Exception as e:
            print("OpenAI error:", e)


    # -----------------------------
    # 2. TRY GEMINI FALLBACK
    # -----------------------------
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            gemini_response = model.generate_content(user_message)
            return gemini_response.text

        except Exception as e:
            print("Gemini error:", e)


    # -----------------------------
    # 3. FINAL DEMO FALLBACK
    # -----------------------------
    return (
        f"[Demo Chatbot] You asked: {user_message}. "
        "Response: Please follow the civic task instructions."
    )
