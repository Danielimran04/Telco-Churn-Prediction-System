"""
============================================================
🌐 Flask Web App — Telco Customer Churn Prediction
============================================================
This is the BACKEND server that:
1. Loads the trained ML model from .pkl files
2. Serves the HTML frontend
3. Receives customer data from the form
4. Runs the prediction and returns the result

HOW TO RUN:
    pip install flask joblib scikit-learn numpy pandas
    python app.py
    Then open http://127.0.0.1:5000 in your browser
============================================================
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# ============================================================
# Initialize Flask App
# ============================================================
app = Flask(__name__)

# ============================================================
# Load the Saved Model, Scaler, and Feature Names
# ============================================================
# 💡 We load these ONCE when the server starts (not on every request)
#    This makes predictions fast since the model is already in memory

MODEL_DIR = 'models'

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'churn_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print("✅ Model, scaler, and feature names loaded successfully!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Number of features: {len(feature_names)}")
except FileNotFoundError:
    print("⚠️  Model files not found! Please run the Jupyter notebook first.")
    print("   Expected files in 'models/' directory:")
    print("   - churn_model.pkl")
    print("   - scaler.pkl")
    print("   - feature_names.pkl")
    model = None
    scaler = None
    feature_names = None


# ============================================================
# Define the Feature Mapping
# ============================================================
# 💡 The form sends human-readable values (e.g., "Yes", "Fiber optic")
#    but our model expects encoded numbers and one-hot columns.
#    This function translates form data → model-ready input.

# These are the RAW columns the user will fill in (before encoding)
RAW_FEATURES = {
    'gender':            ['Female', 'Male'],
    'SeniorCitizen':     ['No', 'Yes'],
    'Partner':           ['No', 'Yes'],
    'Dependents':        ['No', 'Yes'],
    'tenure':            'number',       # 0-72 months
    'PhoneService':      ['No', 'Yes'],
    'MultipleLines':     ['No', 'Yes', 'No phone service'],
    'InternetService':   ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity':    ['No', 'Yes', 'No internet service'],
    'OnlineBackup':      ['No', 'Yes', 'No internet service'],
    'DeviceProtection':  ['No', 'Yes', 'No internet service'],
    'TechSupport':       ['No', 'Yes', 'No internet service'],
    'StreamingTV':       ['No', 'Yes', 'No internet service'],
    'StreamingMovies':   ['No', 'Yes', 'No internet service'],
    'Contract':          ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling':  ['No', 'Yes'],
    'PaymentMethod':     ['Electronic check', 'Mailed check',
                          'Bank transfer (automatic)', 'Credit card (automatic)'],
    'MonthlyCharges':    'number',       # 18.25 - 118.75
    'TotalCharges':      'number',       # 0 - 8684.80
}


def process_input(form_data):
    """
    Convert raw form data into the exact format the model expects.
    
    This is the MOST IMPORTANT function — it must produce a DataFrame
    with the EXACT same columns (in the same order) as the training data.
    
    Steps:
    1. Create a single-row DataFrame with raw values
    2. Label encode binary columns
    3. One-hot encode multi-category columns
    4. Ensure all expected columns exist (fill missing with 0)
    5. Reorder columns to match training data
    """
    
    # Step 1: Extract values from form
    raw = {
        'gender':           form_data.get('gender', 'Male'),
        'SeniorCitizen':    int(form_data.get('SeniorCitizen', '0')),
        'Partner':          form_data.get('Partner', 'No'),
        'Dependents':       form_data.get('Dependents', 'No'),
        'tenure':           int(form_data.get('tenure', '0')),
        'PhoneService':     form_data.get('PhoneService', 'No'),
        'MultipleLines':    form_data.get('MultipleLines', 'No'),
        'InternetService':  form_data.get('InternetService', 'DSL'),
        'OnlineSecurity':   form_data.get('OnlineSecurity', 'No'),
        'OnlineBackup':     form_data.get('OnlineBackup', 'No'),
        'DeviceProtection': form_data.get('DeviceProtection', 'No'),
        'TechSupport':      form_data.get('TechSupport', 'No'),
        'StreamingTV':      form_data.get('StreamingTV', 'No'),
        'StreamingMovies':  form_data.get('StreamingMovies', 'No'),
        'Contract':         form_data.get('Contract', 'Month-to-month'),
        'PaperlessBilling': form_data.get('PaperlessBilling', 'No'),
        'PaymentMethod':    form_data.get('PaymentMethod', 'Electronic check'),
        'MonthlyCharges':   float(form_data.get('MonthlyCharges', '0')),
        'TotalCharges':     float(form_data.get('TotalCharges', '0')),
    }
    
    # Step 2: Create DataFrame
    df = pd.DataFrame([raw])
    
    # Step 3: Label encode binary columns (same as notebook)
    binary_map = {
        'gender':         {'Female': 0, 'Male': 1},
        'Partner':        {'No': 0, 'Yes': 1},
        'Dependents':     {'No': 0, 'Yes': 1},
        'PhoneService':   {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
    }
    
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping)
    
    # Step 4: One-hot encode multi-category columns
    multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
    
    # Step 5: Ensure ALL expected columns exist
    # (One-hot encoding might not create all columns if the input
    #  doesn't have all possible categories)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Step 6: Reorder columns to match training data EXACTLY
    df = df[feature_names]
    
    # Step 7: Convert all to numeric (safety check)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df


# ============================================================
# Routes
# ============================================================

@app.route('/')
def home():
    """Serve the main prediction page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive customer data, run prediction, return result.
    
    This endpoint:
    1. Gets the JSON data from the frontend form
    2. Processes it into model-ready format
    3. Scales the features using the saved scaler
    4. Runs the prediction
    5. Returns the result as JSON
    """
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please run the Jupyter notebook first to train and save the model.'
        }), 500
    
    try:
        # Get form data
        form_data = request.get_json()
        
        # Process input into model-ready format
        input_df = process_input(form_data)
        
        # Scale the features (MUST use the same scaler from training!)
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Build response
        result = {
            'prediction': int(prediction),
            'churn_label': 'HIGH RISK — Likely to Churn' if prediction == 1 else 'LOW RISK — Likely to Stay',
            'probability_stay': round(float(probability[0]) * 100, 2),
            'probability_churn': round(float(probability[1]) * 100, 2),
            'risk_level': 'high' if probability[1] > 0.7 else ('medium' if probability[1] > 0.4 else 'low'),
            'model_used': type(model).__name__
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None,
        'num_features': len(feature_names) if feature_names else 0
    })


# ============================================================
# Run the Server
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Telco Customer Churn Prediction — Flask Server")
    print("=" * 60)
    print("Open your browser at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, port=5000)
