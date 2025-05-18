import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# You'll overwrite this list in Preprocess, but useful as placeholder
PREDICTORS = []
TARGET = ['isFraud']


def save_model(model, columns, output_dir='../output'):
    """Save trained model and column mapping to disk"""
    model.save(f"{output_dir}/deep-ae-model")
    with open(f"{output_dir}/columns.mapping", 'wb') as f:
        pickle.dump(columns, f)
    return True


def load_model(model_path, output_dir='../output'):
    """Load trained model and column mapping from disk"""
    model = keras.models.load_model(model_path)
    with open(f"{output_dir}/columns.mapping", 'rb') as f:
        columns = pickle.load(f)
    return model, columns


def plot_results(model, history, test_X, test_y, mse, threshold):
    """Plot evaluation results for the trained model"""
    # Create figure with multiple subplots
    plt.figure(figsize=(15, 15))
    
    # 1. Plot training history
    plt.subplot(3, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # 2. Plot reconstruction error distribution for normal transactions
    plt.subplot(3, 2, 2)
    normal_idx = test_y == 0
    plt.hist(mse[normal_idx], bins=50, alpha=0.6, color='green', label='Normal')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error: Normal Transactions')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.legend()
    
    # 3. Plot reconstruction error distribution for fraudulent transactions
    plt.subplot(3, 2, 3)
    fraud_idx = test_y == 1
    plt.hist(mse[fraud_idx], bins=50, alpha=0.6, color='red', label='Fraud')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error: Fraudulent Transactions')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.legend()
    
    # 4. Plot ROC curve
    plt.subplot(3, 2, 4)
    fpr, tpr, _ = roc_curve(test_y, mse)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # 5. Plot Precision-Recall curve
    plt.subplot(3, 2, 5)
    precision, recall, _ = precision_recall_curve(test_y, mse)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # 6. Plot confusion matrix
    plt.subplot(3, 2, 6)
    y_pred = [1 if error > threshold else 0 for error in mse]
    cm = confusion_matrix(test_y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    # Save the plots
    plt.savefig('../output/model_evaluation.png')
    plt.close()
    
    # Print classification metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    print("\nPlots saved to '../output/model_evaluation.png'")