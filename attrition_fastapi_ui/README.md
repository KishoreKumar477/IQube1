# Employee Attrition Predictor — Manual FastAPI Form

This UI predicts attrition for **one employee at a time**. Enter the 30 input fields manually in the browser; no CSV upload is required.

## Setup

Copy `app.py`, the `templates/` folder, and `requirements-fastapi.txt` into the root of your existing `IBM_HR_Attrition_Prediction` project. Keep your existing `src/` folder and copy the trained model into:

`models/LogisticRegression_SMOTE_only.joblib`

From the project root, activate your existing virtual environment and install the UI dependencies:

```bash
pip install -r requirements-fastapi.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

The form includes all 30 original model input fields. It also applies the same engineered features from `src/preprocessing.py` and uses the model's 0.60 classification threshold. Example values are prefilled so the page can be tested immediately; replace them with the employee's actual values before relying on a prediction.
