# llm_client.py

import random
import os

# Optional OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from civismart_ai.prompt_templates import LLM_FEATURE_PROMPT

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
        openai.api_key = OPENAI_API_KEY
        try:
            # Using new v1 API
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI that extracts numeric features (0-1) from civic reports."},
                    {"role": "user", "content": LLM_FEATURE_PROMPT.format(report_text=report_text)}
                ]
            )
            import json
            features = json.loads(response.choices[0].message.content)
        except Exception:
            # fallback if parsing fails
            features = _simulate_features()
    else:
        # Use simulated features for MVP/demo
        features = _simulate_features()
    return features

def _simulate_features():
    """Generate random features (0-1) for MVP/demo"""
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
