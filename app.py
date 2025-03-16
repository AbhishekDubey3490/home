from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('house_price_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request (JSON format)
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)  # Reshape for prediction
        
        # Predict using the loaded model
        prediction = model.predict(features)
        
        return jsonify({'predicted_price': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
