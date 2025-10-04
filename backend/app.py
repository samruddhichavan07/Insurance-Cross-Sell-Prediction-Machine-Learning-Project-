from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import json
import os
from sklearn.calibration import CalibratedClassifierCV

# ------------------------------
# Create Flask app
# ------------------------------
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# ------------------------------
# Load artifacts
# ------------------------------
FEATURES_PATH = "artifacts/feature_names.json"
FEATURES = [
    'Gender_Male', 'Gender_Female', 'Age_log', 'Driving_License',
    'Region_Code_Encoding', 'Previously_Insured', 'Vehicle_Age_Encoding',
    'Vehicle_Damage_Encoding', 'Annual_Premium', 'Policy_Sales_Channel_Encoding', 'Vintage'
]

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "r") as f:
        FEATURES = json.load(f)

# Load scaler
scaler_path = "artifacts/scaler.joblib"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError("Scaler not found at artifacts/scaler.joblib")

# Load model
model_path = "artifacts/model_xgb.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError("Model not found at artifacts/model_xgb.joblib")

# Load calibration dataset if available
cal_X_path = "artifacts/X_train_scaled.joblib"
cal_y_path = "artifacts/y_train.joblib"
if os.path.exists(cal_X_path) and os.path.exists(cal_y_path):
    X_train_scaled = joblib.load(cal_X_path)
    y_train = joblib.load(cal_y_path)
    calibrator = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
    calibrator.fit(X_train_scaled, y_train)
else:
    calibrator = model  # fallback

# ------------------------------
# Utility function
# ------------------------------
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # Add missing columns
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0
    df = df.reindex(columns=FEATURES)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # Fix gender encoding
    if 'Gender_Male' in df.columns and 'Gender_Female' in df.columns:
        both_zero = (df['Gender_Male'] == 0) & (df['Gender_Female'] == 0)
        df.loc[both_zero, 'Gender_Female'] = 1
        both_one = (df['Gender_Male'] == 1) & (df['Gender_Female'] == 1)
        df.loc[both_one, 'Gender_Female'] = 0

    return df

# ------------------------------
# Routes
# ------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "features_expected": FEATURES})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid JSON format'}), 400

        df_prepped = prepare_df(df)
        X_input = scaler.transform(df_prepped)

        # Predict probability
        probs = calibrator.predict_proba(X_input)
        pred_label = "YES" if probs[0][1] >= 0.01 else "NO"
        prob_yes = round(float(probs[0][1]), 2)  # 0-1 scale

        return jsonify({
            'used_features': FEATURES,
            'prediction': pred_label,
            'probability': prob_yes
        })

    except Exception as e:
        print("Processing error:", str(e))
        return jsonify({'error': f'Processing error: {str(e)}'}), 400

# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
