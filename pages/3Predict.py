import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(
    page_title='Predict Churn',
    page_icon='',
    layout='wide'
)

# Load encoder and scaler
encoder = joblib.load("C:/Users/hp/OneDrive/Github/Azubi_Capstone/Models/encoder.pkl")
scaler = joblib.load("C:/Users/hp/OneDrive/Github/Azubi_Capstone/Models/scaler.pkl")

# Load trained RandomForestClassifier model
random_forest_model = joblib.load("C:/Users/hp/OneDrive/Github/Azubi_Capstone/Models/random_forest_model.pkl")

# Define mean, min, and max values for float features
float_feature_info = {
    'MONTANT': {'mean': 5529.210895, 'min': 20, 'max': 470000},
    'FREQUENCE_RECH': {'mean': 11.523756, 'min': 1, 'max': 131},
    'REVENUE': {'mean': 5506.050798, 'min': 1, 'max': 532177},
    'ARPU_SEGMENT': {'mean': 1835.355961, 'min': 0, 'max': 177392},
    'FREQUENCE': {'mean': 13.974439, 'min': 1, 'max': 91},
    'DATA_VOLUME': {'mean': 3368.801722, 'min': 0, 'max': 1702309},
    'ON_NET': {'mean': 277.065798, 'min': 0, 'max': 50809},
    'ORANGE': {'mean': 95.160804, 'min': 0, 'max': 12040},
    'TIGO': {'mean': 23.105018, 'min': 0, 'max': 4174},
    'ZONE1': {'mean': 8.167483, 'min': 0, 'max': 2507},
    'ZONE2': {'mean': 7.709844, 'min': 0, 'max': 3697},
    'REGULARITY': {'mean': 28.044496, 'min': 1, 'max': 62},
    'FREQ_TOP_PACK': {'mean': 9.262446, 'min': 1, 'max': 624}
}

# Function to predict churn and calculate probability
def predict_churn(tenure, **float_features):
    # Encoding tenure
    tenure_encoded = encoder.transform([tenure])[0]
    
    # Extract float features and scale them
    features = []
    for feature in float_features:
        if feature in float_feature_info:
            value = float_features[feature]
            min_value = float(float_feature_info[feature]['min'])
            max_value = float(float_feature_info[feature]['max'])
            # Ensure the value is within the specified range
            value = min(max(value, min_value), max_value)
            # Scale the value
            scaled_value = (value - min_value) / (max_value - min_value)
            features.append(scaled_value)
    
    # If there are no numerical features, return a default prediction
    if not features:
        return "Not Churn", 0.0
    
    # Insert encoded 'TENURE' back to features
    features_with_tenure = [tenure_encoded] + features
    
    # Predict churn probabilities
    churn_probabilities = random_forest_model.predict_proba([features_with_tenure])[0]
    
    # Extract probability of churn
    churn_probability = churn_probabilities[1]  # Probability of positive class (churn)
    
    # Predict churn
    churn_prediction = random_forest_model.predict([features_with_tenure])[0]
    
    # Translate churn prediction
    churn_result = "Churn" if churn_prediction == 1 else "Not Churn"
    
    return churn_result, churn_probability

# Define lists of features for each column
features_column_1 = [('MONTANT', 20, 470000), ('FREQUENCE_RECH', 1, 131)]
features_column_2 = [('REVENUE', 1, 532177), ('ARPU_SEGMENT', 0, 177392), ('FREQUENCE', 1, 91)]
features_column_3 = [('DATA_VOLUME', 0, 1702309), ('ON_NET', 0, 50809), ('ORANGE', 0, 12040)]
features_column_4 = [('TIGO', 0, 4174), ('ZONE1', 0, 2507), ('ZONE2', 0, 3697)]
features_column_5 = [('REGULARITY', 1, 62), ('FREQ_TOP_PACK', 1, 624)]

# Streamlit app
def main():
    st.subheader("Churn Prediction App")
    st.write("Enter the following inputs to predict churn. The numbers in brackets are min and max that your values should be in between. The default valuevalues are teh mean values of the training dataset")

    # Arrange features into five columns
    col1, col2, col3, col4, col5 = st.columns(5)

    # Store inputs for history
    inputs = {}

    # First column
    with col1:
        tenure = st.selectbox("TENURE", ['K > 24 month', 'G 12-15 month', 'J 21-24 month', 'H 15-18 month',
                                         'I 18-21 month', 'D 3-6 month', 'F 9-12 month', 'E 6-9 month'])
        for feature, min_val, max_val in features_column_1:
            inputs[feature] = st.number_input(f"{feature} [{min_val}-{max_val}]", 
                                              value=float(float_feature_info[feature]['mean']), 
                                              min_value=float(min_val), max_value=float(max_val))

    # Second column
    with col2:
        for feature, min_val, max_val in features_column_2:
            inputs[feature] = st.number_input(f"{feature} [{min_val}-{max_val}]", 
                                              value=float(float_feature_info[feature]['mean']), 
                                              min_value=float(min_val), max_value=float(max_val))

    # Third column
    with col3:
        for feature, min_val, max_val in features_column_3:
            inputs[feature] = st.number_input(f"{feature} [{min_val}-{max_val}]", 
                                              value=float(float_feature_info[feature]['mean']), 
                                              min_value=float(min_val), max_value=float(max_val))
    # Fourth column
    with col4:
        for feature, min_val, max_val in features_column_4:
            inputs[feature] = st.number_input(f"{feature} [{min_val}-{max_val}]", 
                                              value=float(float_feature_info[feature]['mean']), 
                                              min_value=float(min_val), max_value=float(max_val))
    # Fifth column
    with col5:
        for feature, min_val, max_val in features_column_5:
            inputs[feature] = st.number_input(f"{feature} [{min_val}-{max_val}]", 
                                              value=float(float_feature_info[feature]['mean']), 
                                              min_value=float(min_val), max_value=float(max_val))

    if st.button("Predict"):
        churn_result, churn_probability = predict_churn(tenure, **inputs)
        st.write(f"The customer will {churn_result} and the probability of churn is {churn_probability*100:.2f}%")


        # Include 'tenure' in the inputs
        inputs['tenure'] = tenure
        
        # Save inputs and predictions to CSV
        df_history = pd.DataFrame({**inputs, 'Churn': [churn_result], 'Probability': [churn_probability]})
        mode = 'a' if os.path.exists("history.csv") else 'w'  # Check if file exists
        df_history.to_csv("history.csv", mode=mode, index=False, header=not os.path.exists("history.csv"))

if __name__ == "__main__":
    main()
