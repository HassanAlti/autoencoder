from flask import Flask, request, jsonify
import pandas as pd
import json
import pickle
from ML_Pipeline.Utils import load_model
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Predict import init as predict_init

app = Flask(__name__)

# load model and threshold once
model, columns = load_model('../output/deep-ae-model')

# Load the threshold
with open('../output/threshold.pkl', 'rb') as f:
    threshold = pickle.load(f)

@app.post('/get_fraud_score')
def score():
    data = json.loads(request.data)
    df = pd.DataFrame([data])
    processed = apply(df, is_train=False)
    score = predict_init(processed, model, columns)
    is_fraud = score > threshold
    return jsonify({
        'score': float(score),
        'threshold': float(threshold),
        'is_fraud': bool(is_fraud)
    })

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5001)