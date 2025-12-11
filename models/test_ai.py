from models.model import CiviSmartAI

# 1. Initialize with dataset

ai = CiviSmartAI(dataset_csv="data/civismart_nigeria_dataset.csv")

# 2. Train the model
print("\n=== TRAINING MODEL ===")
ai.train()

# 3. Test prediction with simulated LLM (no API key needed)
print("\n=== PREDICTION TEST (SIMULATED FEATURES) ===")
report = "There is a fire outbreak near the Abuja market and people are rushing out."
result = ai.predict(report_text=report, use_openai=False)
print(result)

# 4. OPTIONAL — Use OpenAI LLM (requires OPENAI_API_KEY)
print("\n=== PREDICTION TEST (REAL OPENAI LLM) ===")
try:
    result2 = ai.predict(report_text=report, use_openai=True)
    print(result2)
except Exception as e:
    print("OpenAI test skipped:", e)
