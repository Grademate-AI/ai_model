from llm_client import call_llm_extract_features
from utils import assign_urgency

# Example report
report_text = "Flooding on Lekki street, houses affected, people trapped"

# 1️⃣ Extract features
features = call_llm_extract_features(report_text, use_openai=False)  # MVP mode
print("Extracted Features (MVP):")
print(features)

# 2️⃣ Assign urgency
urgency_label = assign_urgency(features)
print("\nAssigned Urgency:", urgency_label)
#Test with  OpenAI
features = call_llm_extract_features(report_text, use_openai=True)
print("Extracted Features (OpenAI):")
print(features)
