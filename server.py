from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil, os
from models.model import CiviSmartAI
from civismart_ai.chatbot import get_chatbot_response

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

@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    os.makedirs("data", exist_ok=True)
    temp_path = f"data/temp_retrain_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        ai.retrain_with_new_data(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")
    finally:
        os.remove(temp_path)

    return {"detail": "Model retrained successfully."}

@app.post("/chat")
def chat(request: ChatRequest):
    response = get_chatbot_response(request.user_message)
    return {"response": response}
#done