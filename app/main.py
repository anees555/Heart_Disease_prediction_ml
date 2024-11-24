from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from pandas.api.types import CategoricalDtype
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific domains for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the saved model pipeline
model = joblib.load('app/decision_tree_model1.pkl')

# Define input schema
class HeartDiseaseInput(BaseModel):
    age: int
    sex: str
    chest_pain_type: str
    resting_blood_pressure: int
    cholestoral: int
    fasting_blood_sugar: str
    rest_ecg: str
    Max_heart_rate: int
    exercise_induced_angina: str
    oldpeak: float
    slope: str
    vessels_colored_by_flourosopy: str
    thalassemia: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API!"}


# Preprocessing function
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # One-hot encoding for categorical variables that are strings
    df = pd.get_dummies(df, columns=['sex', 'fasting_blood_sugar', 'exercise_induced_angina'], 
                         prefix=['sex', 'fasting_blood_sugar', 'exercise_induced_angina'], drop_first=True, dtype=int)
    
    expected_columns = ['sex_male', 'fasting_blood_sugar_True', 'exercise_induced_angina_Yes', 'chest_pain_type', 
                        'rest_ecg', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia', 'age', 'resting_blood_pressure',
                        'cholestoral', 'Max_heart_rate', 'oldpeak']
    
    # Encoding 'chest_pain_type' with categorical mapping
    chest_pain_mapping = {
        "Typical angina": 1,
        "Atypical angina": 2,
        "Non-anginal pain": 3,
        "Asymptomatic": 4
    }
    df["chest_pain_type"] = df["chest_pain_type"].map(chest_pain_mapping).fillna(0)

    # Encoding 'rest_ecg' using categorical mapping
    rest_ecg_mapp = CategoricalDtype(categories=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], ordered=True)
    df["rest_ecg"] = df["rest_ecg"].astype(rest_ecg_mapp).cat.codes
    
    # Encoding 'slope' using categorical mapping
    slope_mapp = CategoricalDtype(categories=["Upsloping", "Flat", "Downsloping"], ordered=True)
    df["slope"] = df["slope"].astype(slope_mapp).cat.codes + 1
    
    # Encoding 'vessels_colored_by_flourosopy' using categorical mapping
    vessel_mapp = CategoricalDtype(categories=["Zero", "One", "Two", "Three", "Four"], ordered=True)
    df["vessels_colored_by_flourosopy"] = df["vessels_colored_by_flourosopy"].astype(vessel_mapp).cat.codes
    
    # Encoding 'thalassemia' with custom mapping
    thal_mapp = {
        "No": 0,
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversable Defect": 7
    }
    df["thalassemia"] = df["thalassemia"].map(thal_mapp)
    
    # Add missing columns with default value (0)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure columns are in the expected order
    return df[expected_columns]

# Define prediction endpoint
@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    # Convert input into the model's expected format
    input_dict = input_data.dict()
    preprocessed_data = preprocess_input(input_dict)

    # Make a prediction
    prediction = model.predict(preprocessed_data)

    print(f"Prediction: {prediction[0]}")

    # Return the prediction
    return {"prediction": int(prediction[0])}
