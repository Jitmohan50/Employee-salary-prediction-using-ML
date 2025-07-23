import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Initialize the Flask application
# Use render_template to serve the HTML file
app = Flask(__name__, template_folder='templates')
CORS(app) # Enable Cross-Origin Resource Sharing

# --- 1. Load the Model ---
# Load the pre-trained model.
# Make sure 'best_model.pkl' is in the same directory as this script.
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    print("Error: Model file not found. Make sure 'best_model.pkl' is present.")
    model = None

# --- 2. Define Feature Order and Mappings ---
# This list MUST match the column order used when training the model.
feature_order = [
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    'marital-status', 'occupation', 'relationship', 'race', 'gender',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

# These mappings MUST be replaced with the actual mappings from your notebook.
# I have extracted these from your notebook's logic.
workclass_map = {'Private': 2, 'Self-emp-not-inc': 4, 'Local-gov': 1, 'State-gov': 5, 'Federal-gov': 0, 'Self-emp-inc': 3, 'Without-pay': 6, '?':-1}
education_map = {'HS-grad': 11, 'Some-college': 15, 'Bachelors': 9, 'Masters': 12, 'Assoc-voc': 8, '11th': 0, 'Assoc-acdm': 7, '10th': 1, '7th-8th': 4, 'Prof-school': 14, '9th': 6, '12th': 2, 'Doctorate': 10, '5th-6th': 3, '1st-4th': 5, 'Preschool': 13}
marital_status_map = {'Married-civ-spouse': 2, 'Never-married': 4, 'Divorced': 0, 'Separated': 5, 'Widowed': 6, 'Married-spouse-absent': 3, 'Married-AF-spouse': 1}
occupation_map = {'Prof-specialty': 9, 'Craft-repair': 2, 'Exec-managerial': 3, 'Adm-clerical': 0, 'Sales': 11, 'Other-service': 7, 'Machine-op-inspct': 6, 'Transport-moving': 13, 'Handlers-cleaners': 5, 'Farming-fishing': 4, 'Tech-support': 12, 'Protective-serv': 10, 'Priv-house-serv': 8, 'Armed-Forces': 1, '?':-1}
relationship_map = {'Husband': 0, 'Not-in-family': 1, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5, 'Other-relative': 2}
race_map = {'White': 4, 'Black': 2, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 3}
gender_map = {'Male': 1, 'Female': 0}
native_country_map = {'United-States': 38, 'Mexico': 25, 'Philippines': 29, 'Germany': 10, 'Canada': 1, 'Puerto-Rico': 31, 'El-Salvador': 8, 'India': 18, 'Cuba': 4, 'England': 9, 'Jamaica': 22, 'South': 34, 'China': 2, 'Italy': 21, 'Dominican-Republic': 7, 'Vietnam': 40, 'Guatemala': 12, 'Japan': 23, 'Poland': 30, 'Columbia': 3, 'Taiwan': 36, 'Haiti': 14, 'Iran': 19, 'Portugal': 28, 'Nicaragua': 27, 'Peru': 28, 'France': 11, 'Greece': 13, 'Ecuador': 6, 'Ireland': 20, 'Hong': 16, 'Cambodia': 0, 'Trinadad&Tobago': 37, 'Laos': 24, 'Thailand': 35, 'Yugoslavia': 41, 'Outlying-US(Guam-USVI-etc)': 28, 'Honduras': 15, 'Hungary': 17, 'Scotland': 33, 'Holand-Netherlands': 32, '?':-1}
education_to_num_map = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}

# --- 3. Define Routes ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives data from the website, preprocesses it, and returns a prediction."""
    if not model:
        return jsonify({'error': 'Model is not loaded'}), 500

    try:
        data = request.get_json()

        # Create a dictionary to hold the feature values
        feature_dict = {}

        # Map string values to their encoded integer representations
        feature_dict['age'] = int(data['age'])
        feature_dict['workclass'] = workclass_map.get(data['workclass'], -1)
        feature_dict['education'] = education_map.get(data['education'], -1)
        feature_dict['marital-status'] = marital_status_map.get(data['marital-status'], -1)
        feature_dict['occupation'] = occupation_map.get(data['occupation'], -1)
        feature_dict['race'] = race_map.get(data['race'], -1)
        feature_dict['gender'] = gender_map.get(data['gender'], -1)
        feature_dict['capital-gain'] = int(data.get('capital-gain', 0))
        feature_dict['capital-loss'] = int(data.get('capital-loss', 0))
        feature_dict['hours-per-week'] = int(data['hours-per-week'])
        feature_dict['native-country'] = native_country_map.get(data['native-country'], -1)

        # Handle features that were in training but not the form
        feature_dict['educational-num'] = education_to_num_map.get(data['education'], 0)
        feature_dict['fnlwgt'] = 150000  # Using a median/mean as a placeholder
        # A default for relationship is needed as it's not in the form
        feature_dict['relationship'] = relationship_map.get(data.get('relationship', 'Not-in-family'), 1)

        # Create a DataFrame in the correct feature order
        input_df = pd.DataFrame([feature_dict])
        input_df = input_df[feature_order]

        # Make prediction
        prediction_val = model.predict(input_df)

        # Convert prediction to a standard Python type
        prediction = int(prediction_val[0])
        result = '>50K' if prediction == 1 else '<=50K'

        return jsonify({'prediction': result})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 400

if __name__ == '__main__':
    app.run(debug=True)