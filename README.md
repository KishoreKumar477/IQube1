# IBM HR Analytics — Employee Attrition Prediction

A machine learning application that predicts employee attrition using the IBM HR Analytics Employee Attrition dataset. The project combines a trained classification model with a **FastAPI web interface** that allows users to manually enter an individual employee's details and receive a prediction.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Model Development](#model-development)
5. [Technology Stack](#technology-stack)
6. [Application Features](#application-features)
7. [Project Structure](#project-structure)
8. [Installation and Setup](#installation-and-setup)
9. [Running the Application](#running-the-application)
10. [How to Use the Interface](#how-to-use-the-interface)
11. [Prediction Workflow](#prediction-workflow)
12. [Model Evaluation](#model-evaluation)
13. [Troubleshooting](#troubleshooting)
14. [Limitations](#limitations)
15. [Future Improvements](#future-improvements)

## 1. Project Overview

Employee attrition refers to employees leaving an organization voluntarily or otherwise. Understanding the factors associated with attrition can help organizations investigate workforce trends and develop employee retention strategies.

This project applies supervised machine learning to the IBM HR Analytics Employee Attrition dataset to estimate whether an employee belongs to the attrition or non-attrition class.

The project includes a browser-based interface built using FastAPI and HTML. Instead of uploading an entire CSV file, users enter the attributes of a single employee through a form. The submitted values are processed using the same feature-engineering and preprocessing logic used during model development, then passed to the trained model for prediction.

The application is intended for learning, experimentation, and demonstration of an end-to-end machine learning workflow.

## 2. Objectives

- Develop a binary classification model for employee attrition prediction.
- Explore approaches to handling an imbalanced classification dataset.
- Compare different model-training and resampling strategies.
- Save the selected trained model for reuse.
- Build a FastAPI application that accepts individual employee records.
- Integrate model inference into a simple browser-based interface.
- Display the predicted class and estimated attrition probability.

## 3. Dataset

**Dataset:** IBM HR Analytics Employee Attrition & Performance

The dataset contains employee-related information, including demographics, job characteristics, compensation, job satisfaction, and work experience.

| Property | Description |
|---|---|
| Total records | 1,470 employees |
| Original columns | 35 |
| Prediction target | `Attrition` |
| Problem type | Binary classification |
| Positive class | `Yes` |
| Negative class | `No` |

### Target distribution

| Attrition class | Records | Percentage |
|---|---:|---:|
| No | 1,233 | 83.88% |
| Yes | 237 | 16.12% |

The dataset is imbalanced because substantially fewer employees belong to the attrition class. This makes accuracy alone insufficient for evaluating model performance.

### Feature categories

The input variables cover areas such as:

- **Personal information:** Age, Gender, MaritalStatus, Education and DistanceFromHome.
- **Employment details:** Department, JobRole, JobLevel and BusinessTravel.
- **Compensation:** MonthlyIncome, HourlyRate, DailyRate, MonthlyRate and PercentSalaryHike.
- **Work experience:** TotalWorkingYears, YearsAtCompany, YearsInCurrentRole and YearsWithCurrManager.
- **Job conditions:** JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction and WorkLifeBalance.
- **Other factors:** OverTime, StockOptionLevel, TrainingTimesLastYear and YearsSinceLastPromotion.

The original dataset contains the target column `Attrition`. It is not required as an input when predicting a new employee's attrition.

## 4. Model Development

The project uses a structured machine learning workflow that includes data preparation, feature engineering, preprocessing, model experimentation, evaluation, and model persistence.

### 4.1 Data preparation

The data preparation process includes:

1. Loading the dataset.
2. Separating the target variable from the input features.
3. Removing columns that do not provide useful predictive information for this implementation.
4. Engineering additional career-related features.
5. Splitting the data into training and testing sets using stratification.

The train-test split uses an 80:20 ratio with `random_state=42`, preserving the approximate class distribution in both sets.

### 4.2 Feature engineering

The preprocessing module creates four additional features:

| Feature | Calculation |
|---|---|
| `CareerYearsOutsideCompany` | `TotalWorkingYears - YearsAtCompany`, clipped at zero |
| `RoleTenureRatio` | `YearsInCurrentRole / YearsAtCompany` |
| `PromotionDelayRatio` | `YearsSinceLastPromotion / YearsAtCompany` |
| `ManagerTenureRatio` | `YearsWithCurrManager / YearsAtCompany` |

A denominator adjustment is used when `YearsAtCompany` is zero to avoid division-by-zero errors.

These features represent career experience outside the current organization and relative tenure measures.

### 4.3 Columns removed

The following columns are excluded from model input:

- `EmployeeNumber`
- `EmployeeCount`
- `Over18`
- `StandardHours`

The target column `Attrition` is separated from the predictors during training and excluded from new-record inference.

### 4.4 Preprocessing

The preprocessing logic is implemented in `src/preprocessing.py`.

Numerical features use:
- Median imputation for missing values.
- Standard scaling.

Categorical features use:
- Most-frequent-value imputation.
- One-hot encoding with `handle_unknown="ignore"`.

The preprocessing configuration is saved as part of the trained model pipeline. This ensures that incoming employee records undergo the transformations expected by the model.

### 4.5 Handling class imbalance

Several approaches were explored during model development, including baseline classifiers, SMOTE, class weighting, and alternative resampling methods.

The selected implementation uses **Logistic Regression with SMOTE**, without additional class weighting.

SMOTE generates synthetic examples of the minority class in the training data to reduce the effect of the original class imbalance. It must be applied only within the training process, not to the held-out test set or new inference records.

### 4.6 Selected model

The current application uses the saved model artifact:

`models/LogisticRegression_SMOTE_only.joblib`

The model was selected based on the observed evaluation results from the experiments conducted in this project. Its decision threshold is configured to `0.60`.

The saved artifact is loaded for inference; the model is not retrained each time the application starts.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and machine learning implementation |
| pandas | Tabular data manipulation |
| NumPy | Numerical operations |
| scikit-learn | Preprocessing, Logistic Regression and evaluation |
| imbalanced-learn | SMOTE resampling |
| joblib | Saving and loading the trained model |
| FastAPI | Web application and prediction endpoint |
| Uvicorn | ASGI server |
| Jinja2 | Rendering the HTML interface |
| HTML and CSS | Form layout and styling |
| python-multipart | Parsing submitted HTML form data |

## 6. Application Features

### Manual employee data entry

The interface accepts one employee record at a time. Users can enter the employee's details directly into the browser instead of preparing and uploading a CSV file.

### Input fields

The form provides 30 model input fields, using numerical inputs for numeric variables and dropdowns for categorical variables.

### Prediction result

After submission, the application displays the predicted attrition class and, where implemented, the model's estimated probability of attrition.

### Reusable model

The application loads a previously trained model from disk. Predictions can be made without repeating the training workflow.

### Browser-based access

The interface runs locally and can be opened using a web browser.

## 7. Project Structure

The relevant project files are organized as follows:

```text
IBM-HR-Attrition-Prediction/
│
├── app.py
├── requirements-fastapi.txt
│
├── templates/
│   └── index.html
│
├── src/
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── threshold_analysis.py
│
├── models/
│   └── LogisticRegression_SMOTE_only.joblib
│
├── reports/
│   ├── lr_smote_only_results.csv
│   └── lr_smote_only_thresholds.csv
│
└── README.md
```

*Note: This is a representative structure. Keep the existing files in your project, and adjust filenames in this README if your actual directory differs.*

### Important files

- `app.py`: Loads the trained model, accepts form submissions, prepares the input record, and returns prediction results.
- `templates/index.html`: Contains the employee input form and the results interface.
- `src/preprocessing.py`: Defines the feature engineering and preprocessing logic.
- `models/LogisticRegression_SMOTE_only.joblib`: Saved model artifact required by the application.
- `requirements-fastapi.txt`: Lists the Python packages needed to run the web interface.
- `reports/`: Stores selected experiment results and threshold-analysis outputs.

## 8. Installation and Setup

### Prerequisites

- Python installed on your system.
- A terminal and a web browser.
- The project source files.
- The trained model artifact at `models/LogisticRegression_SMOTE_only.joblib`.

### Step 1: Open the project directory

Navigate to the project root, which should contain `app.py`, `src/`, and `models/`.

For example, if the project is stored on your Desktop:

```bash
cd "$HOME/Desktop/IQube!"
```

Use your actual project directory if it is stored elsewhere.

### Step 2: Activate the virtual environment

If you already have a virtual environment named `.venv`, activate it:

```bash
source .venv/bin/activate
```

On macOS, the terminal should indicate that the virtual environment is active.

### Step 3: Install dependencies

Install the packages listed in the FastAPI requirements file:

```bash
python -m pip install -r requirements-fastapi.txt
```

Make sure `requirements-fastapi.txt` includes `python-multipart`. It is required for parsing the submitted HTML form.

If it is missing, install it directly:

```bash
python -m pip install python-multipart
```

## 9. Running the Application

From the project root, run:

```bash
uvicorn app:app --reload
```

When the server starts successfully, open:

**http://127.0.0.1:8000**

To stop the server, press `Ctrl + C` in the terminal.

### Expected startup behavior

Uvicorn should report that the application has started successfully. Opening the URL should load the HTML form. Submitting a valid employee record should send a request to the prediction endpoint.

The `--reload` option automatically restarts the development server when source files change. It is intended for local development rather than production deployment.

## 10. How to Use the Interface

1. Start the FastAPI application.
2. Open the local application URL in your browser.
3. Enter the employee's information in the available fields.
4. Select the appropriate categorical values from the dropdowns.
5. Enter the numerical values in the corresponding fields.
6. Submit the form using the prediction button.
7. Review the predicted attrition class and probability displayed by the interface.

Example values may be supplied by the form for demonstration. Replace them with the record you want to evaluate.

The model expects input values that follow the dataset's feature definitions and categories. Arbitrary values or inconsistent employee details can produce misleading predictions.

## 11. Prediction Workflow

The application follows this inference workflow:

```text
User opens the web interface
            |
            v
Enters one employee's details
            |
            v
Submits the HTML form
            |
            v
FastAPI receives the form data
            |
            v
Input values are converted to a record
            |
            v
Feature engineering is applied
            |
            v
Trained preprocessing pipeline transforms data
            |
            v
Logistic Regression estimates attrition probability
            |
            v
Probability is compared with threshold 0.60
            |
            v
Prediction result is returned to the interface
```

### Decision threshold

The application uses a threshold of `0.60`:

- If the estimated probability is greater than or equal to `0.60`, the predicted class is `Yes`.
- If the estimated probability is below `0.60`, the predicted class is `No`.

This threshold changes the trade-off between precision and recall. It is not a measure of certainty and does not mean that the model is 60% accurate.

## 12. Model Evaluation

The selected Logistic Regression model with SMOTE was evaluated on the held-out test set.

| Metric | Observed result |
|---|---:|
| Accuracy | 0.8435 |
| Precision | 0.5094 |
| Recall | 0.5745 |
| F1-score | 0.5400 |
| ROC-AUC | 0.8259 |
| Average precision | 0.5873 |

These figures correspond to the model evaluated at a threshold of `0.50`, not the application's configured threshold of `0.60`.

At the `0.60` threshold selected using training out-of-fold predictions, the observed test results were:

| Metric | Observed result |
|---|---:|
| Accuracy | 0.8605 |
| Precision | 0.5682 |
| Recall | 0.5319 |
| F1-score | 0.5495 |

### Interpretation

- **Accuracy** measures the overall proportion of correct predictions.
- **Precision** measures the proportion of predicted attrition cases that were actually attrition cases.
- **Recall** measures the proportion of actual attrition cases identified by the model.
- **F1-score** combines precision and recall into a single metric.
- **ROC-AUC** measures the model's ability to rank positive cases above negative cases across classification thresholds.
- **Average precision** summarizes precision-recall performance.

The results indicate that the model identifies some employees in the attrition class while maintaining a balance between false positives and false negatives. However, precision and recall are not high enough to justify treating the predictions as definitive outcomes.

The reported metrics are observations from the current experimental setup, not guarantees of performance on other organizations or future employee populations.

## 13. Troubleshooting

### Error: `The python-multipart library must be installed to use form parsing`

**Cause:** The application uses `request.form()`, which requires `python-multipart`.

**Solution:**

```bash
python -m pip install python-multipart
```

Restart the server after installation.

### Error: `SyntaxError: invalid syntax` near `MODEL_PATH =`

**Cause:** The model path assignment is incomplete or contains invalid Python syntax.

**Solution:** Ensure the assignment contains a valid path, for example:

```python
MODEL_PATH = "models/LogisticRegression_SMOTE_only.joblib"
```

The exact path must match the location of the saved model.

### Error: model file not found

**Cause:** The trained model is missing, or the application is being started from the wrong working directory.

**Solution:**

- Confirm that `models/LogisticRegression_SMOTE_only.joblib` exists.
- Run Uvicorn from the project root.
- Check that `MODEL_PATH` matches the artifact's location.

### Error: `ModuleNotFoundError` for `src.preprocessing`

**Cause:** The existing preprocessing module cannot be imported.

**Solution:**

- Confirm that `src/preprocessing.py` exists.
- Run the command from the project root.
- Activate the correct virtual environment.
- Check that the module exposes the functions used by `app.py`, particularly `engineer_features` and the configured dropped-column list.

### The homepage loads, but prediction returns HTTP 500

A successful `GET /` response only confirms that the page can load. It does not confirm that the prediction workflow is functioning.

Check the terminal traceback after submitting the form. Verify that:

- All required form fields are present.
- Numerical inputs can be converted to the expected data types.
- The input columns match those expected by the trained model.
- The preprocessing module is compatible with the saved artifact.
- All required dependencies are installed in the active environment.

## 14. Limitations

- The model is trained on a specific historical dataset and may not generalize to other organizations.
- Predictions depend on the quality and accuracy of manually entered information.
- The dataset contains class imbalance, and the model can produce both false-positive and false-negative predictions.
- The selected threshold reflects an experimental trade-off and may not be appropriate for every use case.
- The model estimates patterns associated with attrition; it does not establish why a specific employee might leave.
- Employee-related predictions can raise fairness, privacy, and ethical concerns.
- The application is a local demonstration and is not designed as a production HR decision system.

Predictions should not be used as the sole basis for employment, promotion, compensation, disciplinary, or termination decisions. Any practical application would require appropriate validation, privacy safeguards, fairness assessments, and human oversight.

## 15. Future Improvements

Potential improvements include:

- Adding input validation and clearer error messages.
- Displaying a confusion matrix and other evaluation visualizations.
- Improving the layout and responsiveness of the interface.
- Adding automated tests for feature engineering and prediction.
- Tracking model versions and documenting changes in performance.
- Evaluating probability calibration and fairness across relevant groups.
- Adding secure deployment, logging, and monitoring if the application moves beyond local experimentation.

## Conclusion

This project demonstrates the integration of a supervised machine learning classifier with a FastAPI web application. It covers the workflow from dataset preparation and feature engineering to model evaluation, model persistence, and individual-record inference.

The current implementation provides a practical way to interact with a trained employee attrition model through a browser without uploading a full dataset or retraining the model for every prediction.

**Project status:** Local prototype for machine learning experimentation and demonstration.
