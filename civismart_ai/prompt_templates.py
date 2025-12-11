# prompt_templates.py

LLM_FEATURE_PROMPT = """
You are an AI assistant that reads civic reports and extracts numeric scores (0-1) for each of the following features:
- danger_score
- health_risk
- environment_damage
- infrastructure_damage
- fire_risk
- flooding_indicator
- urgency_keywords_score
- human_harm_risk

Report:
{report_text}

Return the result as a JSON dictionary with each feature as a key and a value between 0 and 1.
"""
