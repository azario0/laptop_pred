import pickle
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model and feature names
def load_model_and_features():
    model = joblib.load('best_model.pkl')
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names, label_encoders

# Load the dataset once to extract column information
df = pd.read_csv('laptop_dataset.csv')  # Ensure you have this file
columns_non_numeric = [col for col in df.drop(columns=['Price (Euro)']).columns if not pd.api.types.is_numeric_dtype(df[col])]
columns_numeric = [col for col in df.drop(columns=['Price (Euro)']).columns if pd.api.types.is_numeric_dtype(df[col])]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_unique_values')
def get_unique_values():
    # Get unique values for non-numeric columns
    unique_values = {col: list(df[col].unique()) for col in columns_non_numeric}
    return jsonify(unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model and preprocessing tools
        model, feature_names, label_encoders = load_model_and_features()
        
        # Get input data
        input_data = request.json
        print("Received input:", input_data)

        # Validate input
        missing_columns = [col for col in feature_names if col not in input_data]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        print(input_df.info())

        # Process non-numeric columns with label encoders
        for col in columns_non_numeric:
            if col in label_encoders:
                mapping = label_encoders[col]
                
                # Map values using the stored dictionary
                input_df[col] = input_df[col].map(mapping)
                
                # Check for unmapped values and handle them if necessary
                if input_df[col].isnull().any():
                    print(f"Warning: Unmapped values found in column '{col}'. These have been replaced with NaN.")
            else:
                raise KeyError(f"No mapping found for column '{col}' in label_encoders.")

        # Ensure numeric columns are properly converted
        for col in columns_numeric:
            if col in input_df:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')


        # Handle missing or invalid numeric data
        if input_df[columns_numeric].isnull().any().any():
            return jsonify({'error': 'Invalid numeric inputs. Please ensure all numeric fields are filled correctly.'}), 400

        # Predict
        prediction = model.predict(input_df[feature_names])
        return jsonify({'predicted_price': float(prediction[0])})

    except Exception as e:
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
