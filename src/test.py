import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.metrics import confusion_matrix
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Predict import init as predict_init

# Configuration
INPUT_PATH = '../input/paysim.csv'
CHUNK_SIZE = 500_000
MODEL_DIR = '../output'
THRESHOLD_PATH = '../output/threshold.pkl'
CUSTOM_THRESHOLD = 0.62

# Define custom loss for autoencoders
@tf.keras.utils.register_keras_serializable()
def custom_mse_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return mse

# Load trained models and column mapping
print("Loading models...")
# Load autoencoders with custom loss
autoencoder1 = keras_load_model(f"{MODEL_DIR}/autoencoder1", custom_objects={'custom_mse_loss': custom_mse_loss})
autoencoder2 = keras_load_model(f"{MODEL_DIR}/autoencoder2", custom_objects={'custom_mse_loss': custom_mse_loss})
# Load transformer without special objects
transformer = keras_load_model(f"{MODEL_DIR}/transformer")

# Load column mapping
with open(f"{MODEL_DIR}/columns.mapping", 'rb') as f:
    columns = pickle.load(f)
print("Models loaded successfully.")

# Load saved threshold
with open(THRESHOLD_PATH, 'rb') as f:
    saved_threshold = pickle.load(f)
print(f"Loaded saved threshold: {saved_threshold:.6f}")
print(f"Custom threshold: {CUSTOM_THRESHOLD:.2f}\n")

# Prepare model dict
model_dict = {
    'autoencoder1': autoencoder1,
    'autoencoder2': autoencoder2,
    'transformer': transformer
}

# Collect true labels and fraud scores
y_true_list = []
scores_list = []

# Iterate over data in chunks
for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
    print(f"Processing chunk {i+1} (rows {i*CHUNK_SIZE} to {(i+1)*CHUNK_SIZE})...")
    # Preprocess chunk
    processed = apply(chunk, is_train=False)
    # Extract true labels
    y_true = processed['isFraud'].values
    # Prepare features
    X_proc = processed.drop('isFraud', axis=1)

    # Get fraud scores
    scores = predict_init(X_proc, model_dict, columns)

    # Accumulate
    y_true_list.append(y_true)
    scores_list.append(scores)

# Concatenate all
y_true_all = np.concatenate(y_true_list)
scores_all = np.concatenate(scores_list)

# Evaluate using thresholds

def eval_confusion(y_true, scores, thr):
    y_pred = (scores > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

# Saved threshold
tn_s, fp_s, fn_s, tp_s = eval_confusion(y_true_all, scores_all, saved_threshold)
# Custom threshold
tn_c, fp_c, fn_c, tp_c = eval_confusion(y_true_all, scores_all, CUSTOM_THRESHOLD)

# Print results
print("Confusion Matrix at Saved Threshold:")
print(f"TN: {tn_s}, FP: {fp_s}, FN: {fn_s}, TP: {tp_s}\n")

print(f"Confusion Matrix at Custom Threshold ({CUSTOM_THRESHOLD}):")
print(f"TN: {tn_c}, FP: {fp_c}, FN: {fn_c}, TP: {tp_c}")