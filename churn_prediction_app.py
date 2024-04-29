from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Get the current directory
current_directory = os.path.dirname(__file__)

# Load trained models and label encoder
catboost_model = joblib.load(os.path.join(current_directory, "catboost_model.pkl"))
decision_tree_model = joblib.load(os.path.join(current_directory, "decision_tree_model.pkl"))
random_forest_model = joblib.load(os.path.join(current_directory, "random_forest_model.pkl"))
xgboost_model = joblib.load(os.path.join(current_directory, "xgboost_model.pkl"))
label_encoder = joblib.load(os.path.join(current_directory, "encoder.pkl"))
scaler = joblib.load(os.path.join(current_directory, "scaler.pkl"))  # Assuming you have saved RobustScaler

class Features(BaseModel):
    TENURE: str
    MONTANT: float
    FREQUENCE_RECH: float
    REVENUE: float
    ARPU_SEGMENT: float
    FREQUENCE: float
    DATA_VOLUME: float
    ON_NET: float
    ORANGE: float
    TIGO: float
    ZONE1: float
    ZONE2: float
    REGULARITY: float
    FREQ_TOP_PACK: float

@app.get("/")
async def root():
    return {"message": "Welcome to Churn Prediction Application"}

@app.post("/predict_churn")
async def predict_churn(features: Features):
    # Encode TENURE using the loaded label encoder
    encoded_tenure = label_encoder.transform([features.TENURE])[0]

    # Convert features to numpy array
    feature_values = np.array([[
        encoded_tenure, features.MONTANT, features.FREQUENCE_RECH, features.REVENUE, features.ARPU_SEGMENT,
        features.FREQUENCE, features.DATA_VOLUME, features.ON_NET, features.ORANGE,
        features.TIGO, features.ZONE1, features.ZONE2, features.REGULARITY, features.FREQ_TOP_PACK
    ]])

    # Normalize features using RobustScaler
    normalized_features = scaler.transform(feature_values)

    # Predict churn using the models
    catboost_prediction = catboost_model.predict(normalized_features)[0]
    decision_tree_prediction = decision_tree_model.predict(normalized_features)[0]
    random_forest_prediction = random_forest_model.predict(normalized_features)[0]
    xgboost_prediction = xgboost_model.predict(normalized_features)[0]

    # Decode churn prediction using LabelEncoder
    decoded_catboost = label_encoder.inverse_transform([catboost_prediction])[0]
    decoded_decision_tree = label_encoder.inverse_transform([decision_tree_prediction])[0]
    decoded_random_forest = label_encoder.inverse_transform([random_forest_prediction])[0]
    decoded_xgboost = label_encoder.inverse_transform([xgboost_prediction])[0]

    return {
        "CatBoost Prediction": decoded_catboost,
        "Decision Tree Prediction": decoded_decision_tree,
        "Random Forest Prediction": decoded_random_forest,
        "XGBoost Prediction": decoded_xgboost
    }
