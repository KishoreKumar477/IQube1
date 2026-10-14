from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.preprocessing import DROP_COLUMNS, TARGET, engineer_features

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "LogisticRegression_SMOTE_only.joblib"
THRESHOLD = 0.60

app = FastAPI(title="Employee Attrition Predictor")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
_model = None

# The form contains the 30 original input features used by the IBM HR dataset,
# excluding Attrition (the value being predicted) and the four dropped columns.
FIELDS = [
    {"name": "Age", "label": "Age", "type": "number", "min": 18, "max": 100, "step": 1, "default": 35, "section": "Personal details"},
    {"name": "BusinessTravel", "label": "Business travel", "type": "select", "options": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], "default": "Travel_Rarely", "section": "Work details"},
    {"name": "DailyRate", "label": "Daily rate", "type": "number", "min": 0, "max": 2000, "step": 1, "default": 800, "section": "Compensation"},
    {"name": "Department", "label": "Department", "type": "select", "options": ["Research & Development", "Sales", "Human Resources"], "default": "Research & Development", "section": "Work details"},
    {"name": "DistanceFromHome", "label": "Distance from home", "type": "number", "min": 0, "max": 1000, "step": 1, "default": 5, "section": "Personal details"},
    {"name": "Education", "label": "Education level (1–5)", "type": "number", "min": 1, "max": 5, "step": 1, "default": 3, "section": "Personal details"},
    {"name": "EducationField", "label": "Education field", "type": "select", "options": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"], "default": "Life Sciences", "section": "Personal details"},
    {"name": "EnvironmentSatisfaction", "label": "Environment satisfaction (1–4)", "type": "number", "min": 1, "max": 4, "step": 1, "default": 3, "section": "Satisfaction and performance"},
    {"name": "Gender", "label": "Gender", "type": "select", "options": ["Male", "Female"], "default": "Male", "section": "Personal details"},
    {"name": "HourlyRate", "label": "Hourly rate", "type": "number", "min": 0, "max": 200, "step": 1, "default": 65, "section": "Compensation"},
    {"name": "JobInvolvement", "label": "Job involvement (1–4)", "type": "number", "min": 1, "max": 4, "step": 1, "default": 3, "section": "Satisfaction and performance"},
    {"name": "JobLevel", "label": "Job level (1–5)", "type": "number", "min": 1, "max": 5, "step": 1, "default": 2, "section": "Work details"},
    {"name": "JobRole", "label": "Job role", "type": "select", "options": ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Executive", "Research Director", "Human Resources", "Sales Representative"], "default": "Research Scientist", "section": "Work details"},
    {"name": "JobSatisfaction", "label": "Job satisfaction (1–4)", "type": "number", "min": 1, "max": 4, "step": 1, "default": 3, "section": "Satisfaction and performance"},
    {"name": "MaritalStatus", "label": "Marital status", "type": "select", "options": ["Single", "Married", "Divorced"], "default": "Married", "section": "Personal details"},
    {"name": "MonthlyIncome", "label": "Monthly income", "type": "number", "min": 0, "max": 1000000, "step": 1, "default": 5000, "section": "Compensation"},
    {"name": "MonthlyRate", "label": "Monthly rate", "type": "number", "min": 0, "max": 100000, "step": 1, "default": 14000, "section": "Compensation"},
    {"name": "NumCompaniesWorked", "label": "Number of companies worked", "type": "number", "min": 0, "max": 100, "step": 1, "default": 2, "section": "Career history"},
    {"name": "OverTime", "label": "Overtime", "type": "select", "options": ["No", "Yes"], "default": "No", "section": "Work details"},
    {"name": "PercentSalaryHike", "label": "Percent salary hike", "type": "number", "min": 0, "max": 100, "step": 1, "default": 13, "section": "Compensation"},
    {"name": "PerformanceRating", "label": "Performance rating (3–4)", "type": "number", "min": 3, "max": 4, "step": 1, "default": 3, "section": "Satisfaction and performance"},
    {"name": "RelationshipSatisfaction", "label": "Relationship satisfaction (1–4)", "type": "number", "min": 1, "max": 4, "step": 1, "default": 3, "section": "Satisfaction and performance"},
    {"name": "StockOptionLevel", "label": "Stock option level (0–3)", "type": "number", "min": 0, "max": 3, "step": 1, "default": 1, "section": "Compensation"},
    {"name": "TotalWorkingYears", "label": "Total working years", "type": "number", "min": 0, "max": 100, "step": 1, "default": 10, "section": "Career history"},
    {"name": "TrainingTimesLastYear", "label": "Training times last year", "type": "number", "min": 0, "max": 100, "step": 1, "default": 3, "section": "Career history"},
    {"name": "WorkLifeBalance", "label": "Work-life balance (1–4)", "type": "number", "min": 1, "max": 4, "step": 1, "default": 3, "section": "Satisfaction and performance"},
    {"name": "YearsAtCompany", "label": "Years at company", "type": "number", "min": 0, "max": 100, "step": 1, "default": 5, "section": "Career history"},
    {"name": "YearsInCurrentRole", "label": "Years in current role", "type": "number", "min": 0, "max": 100, "step": 1, "default": 3, "section": "Career history"},
    {"name": "YearsSinceLastPromotion", "label": "Years since last promotion", "type": "number", "min": 0, "max": 100, "step": 1, "default": 1, "section": "Career history"},
    {"name": "YearsWithCurrManager", "label": "Years with current manager", "type": "number", "min": 0, "max": 100, "step": 1, "default": 3, "section": "Career history"},
]

SECTIONS = [
    "Personal details",
    "Work details",
    "Career history",
    "Compensation",
    "Satisfaction and performance",
]


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Copy your saved "
                "LogisticRegression_SMOTE_only.joblib into the models/ folder."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def render(request: Request, **context):
    base = {"fields": FIELDS, "sections": SECTIONS, "error": None, "prediction": None, "values": {}}
    base.update(context)
    return templates.TemplateResponse(request=request, name="index.html", context=base)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render(request)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    values = {field["name"]: form.get(field["name"], "") for field in FIELDS}
    try:
        record = {}
        for field in FIELDS:
            name = field["name"]
            raw = values[name]
            if raw is None or str(raw).strip() == "":
                raise ValueError(f"Please fill in {field['label']}.")
            if field["type"] == "number":
                number = float(raw)
                if not number.is_integer() and field["step"] == 1:
                    raise ValueError(f"{field['label']} must be a whole number.")
                number = int(number) if field["step"] == 1 else number
                if number < field["min"] or number > field["max"]:
                    raise ValueError(f"{field['label']} must be between {field['min']} and {field['max']}.")
                record[name] = number
            else:
                if raw not in field["options"]:
                    raise ValueError(f"Select a valid value for {field['label']}.")
                record[name] = raw

        # Add engineered features exactly as in the training workflow.
        df = engineer_features(pd.DataFrame([record]))
        df = df.drop(columns=[TARGET, *DROP_COLUMNS], errors="ignore")
        model = get_model()
        expected = list(model.named_steps["preprocessor"].feature_names_in_)
        missing = [column for column in expected if column not in df.columns]
        if missing:
            raise ValueError("The form does not provide model features: " + ", ".join(missing))
        X = df[expected]
        probability = float(model.predict_proba(X)[0, 1])
        predicted_yes = probability >= THRESHOLD
        prediction = {
            "label": "Yes — elevated attrition risk" if predicted_yes else "No — lower predicted attrition risk",
            "yes": predicted_yes,
            "probability": probability * 100,
            "threshold": THRESHOLD * 100,
        }
        return render(request, values=values, prediction=prediction)
    except Exception as exc:
        return render(request, values=values, error=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH.name, "model_exists": MODEL_PATH.exists()}
