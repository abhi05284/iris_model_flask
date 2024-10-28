from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('iris_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([np.array(data['features'])])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000)
