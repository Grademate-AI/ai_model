# utils.py

def assign_urgency(row):
    """
    Simple threshold-based urgency assignment (Low/Medium/High/Critical)
    based on danger_score, human_harm_risk, flooding_indicator
    """
    score = (row.get("danger_score",0) + row.get("human_harm_risk",0) + row.get("flooding_indicator",0))/3
    if score > 0.7:
        return "Critical"
    elif score > 0.5:
        return "High"
    elif score > 0.3:
        return "Medium"
    else:
        return "Low"
