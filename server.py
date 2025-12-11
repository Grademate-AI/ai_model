from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from models.model import CiviSmartAI
from civismart_ai.chatbot import get_chatbot_response
import sqlalchemy

# -----------------------------
# Point mapping for volunteers
# -----------------------------
POINT_MAPPING = {
    "Critical": 100,
    "High": 70,
    "Medium": 40,
    "Low": 20
}

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="CiviSmart AI API")

# Initialize AI model
DATASET_CSV = "data/civismart_nigeria_dataset.csv"
ai = CiviSmartAI(dataset_csv=DATASET_CSV)
ai.train()  # ensure model is ready

# -----------------------------
# Pydantic Models
# -----------------------------
class Report(BaseModel):
    report_text: str
    use_openai: bool = False
    volunteer_score: float = 1.0
    task_complexity: float = 1.0

class ChatRequest(BaseModel):
    user_message: str

# -----------------------------
# Helper Functions
# -----------------------------
def calculate_points(urgency_label: str, volunteer_score: float = 1.0, task_complexity: float = 1.0) -> float:
    base_points = POINT_MAPPING.get(urgency_label, 0)
    points = base_points * volunteer_score * task_complexity
    return round(points, 0)

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict")
def predict(report: Report):
    try:
        result = ai.predict(report.report_text, use_openai=report.use_openai)
        result["points"] = calculate_points(
            result["urgency"],
            volunteer_score=report.volunteer_score,
            task_complexity=report.task_complexity
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain/{db_url}")
def retrain(db_url: str):
    """
    Retrain the AI model using a database URL.
    db_url should be a valid SQLAlchemy connection string.
    """
    try:
        # Connect to the database
        engine = sqlalchemy.create_engine(db_url)
        # Assume table "verified_reports" has same columns as dataset CSV
        df_new = pd.read_sql_table("verified_reports", con=engine)

        # Combine with existing dataset
        df_old = pd.read_csv(DATASET_CSV)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(DATASET_CSV, index=False)

        # Retrain model
        ai.train()
        return {"detail": f"Model retrained successfully with {len(df_new)} new rows."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = get_chatbot_response(request.user_message, use_openai=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {e}")

#done