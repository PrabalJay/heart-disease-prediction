from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('models/best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'age': float(request.form['age']),
            'sex': int(request.form['sex']),
            'chest pain type': int(request.form['chest_pain']),
            'resting bp s': float(request.form['resting_bp']),
            'cholesterol': float(request.form['cholesterol']),
            'fasting blood sugar': int(request.form['fasting_bs']),
            'resting ecg': int(request.form['resting_ecg']),
            'max heart rate': float(request.form['max_hr']),
            'exercise angina': int(request.form['exercise_angina']),
            'oldpeak': float(request.form['oldpeak']),
            'ST slope': int(request.form['st_slope'])
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(data, index=[0])
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]
        
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability)
        }
        
        return render_template('index.html', 
                             prediction_text=f'Heart Disease: {"Yes" if result["prediction"] else "No"} (Probability: {result["probability"]:.2f})')
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)