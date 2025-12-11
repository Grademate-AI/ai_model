import random
import os

# Optional: for OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def call_llm_extract_features(report_text, use_openai=False):
    """
    Extract numeric features (0-1) from report text.
    Params:
        report_text: str
        use_openai: bool - if True, use OpenAI API
    Returns:
        dict of features
    """
    if use_openai and OPENAI_AVAILABLE and OPENAI_API_KEY:
        # Call OpenAI API
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that extracts numeric features (0-1) from civic reports."},
                {"role": "user",
                 "content": f"Extract the following features from this report (values 0-1): danger_score, health_risk, environment_damage, infrastructure_damage, fire_risk, flooding_indicator, urgency_keywords_score, human_harm_risk.\nReport: {report_text}"}
            ]
        )
        # Parse the response into a dictionary
        # For now, assume GPT returns JSON
        try:
            import json
            features = json.loads(response.choices[0].message.content)
        except:
            # fallback to random if parsing fails
            features = _simulate_features()
    else:
        # Use MVP simulated features
        features = _simulate_features()

    return features


def _simulate_features():
    """Generate random features (0-1)"""
    return {
        "danger_score": round(random.uniform(0, 1), 2),
        "health_risk": round(random.uniform(0, 1), 2),
        "environment_damage": round(random.uniform(0, 1), 2),
        "infrastructure_damage": round(random.uniform(0, 1), 2),
        "fire_risk": round(random.uniform(0, 1), 2),
        "flooding_indicator": round(random.uniform(0, 1), 2),
        "urgency_keywords_score": round(random.uniform(0, 1), 2),
        "human_harm_risk": round(random.uniform(0, 1), 2),
    }
