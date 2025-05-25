# File: src/ML_Pipeline/Utils.py
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import os

def save_model(model_dict, columns, output_dir='../output'):
    """Save BankSafeNet model components and column mapping to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each model component
    model_dict['autoencoder1'].save(f"{output_dir}/autoencoder1")
    model_dict['autoencoder2'].save(f"{output_dir}/autoencoder2")
    model_dict['transformer'].save(f"{output_dir}/transformer")
    
    # Save column mapping
    with open(f"{output_dir}/columns.mapping", 'wb') as f:
        pickle.dump(columns, f)
    
    print("BankSafeNet model components saved successfully!")
    return True

def load_model(model_path, output_dir='../output'):
    """Load BankSafeNet model components and column mapping from disk"""
    model_dict = {
        'autoencoder1': keras.models.load_model(f"{output_dir}/autoencoder1"),
        'autoencoder2': keras.models.load_model(f"{output_dir}/autoencoder2"),
        'transformer': keras.models.load_model(f"{output_dir}/transformer")
    }
    
    with open(f"{output_dir}/columns.mapping", 'rb') as f:
        columns = pickle.load(f)
    
    print("BankSafeNet model components loaded successfully!")
    return model_dict, columns

def plot_results(model_dict, history, test_X, test_y, fraud_scores, threshold, output_dir='../output'):
    """Plot comprehensive evaluation results for BankSafeNet"""
    plt.figure(figsize=(20, 15))
    
    # 1. Autoencoder 1 Training Loss
    plt.subplot(3, 4, 1)
    plt.plot(history['ae1_loss'], label='Train Loss', color='blue')
    plt.plot(history['ae1_val_loss'], label='Val Loss', color='red')
    plt.title('Autoencoder 1 (Normal Patterns) Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. Autoencoder 2 Training Loss
    plt.subplot(3, 4, 2)
    plt.plot(history['ae2_loss'], label='Train Loss', color='blue')
    plt.plot(history['ae2_val_loss'], label='Val Loss', color='red')
    plt.title('Autoencoder 2 (Anomaly Detection) Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 3. Transformer Training Loss
    plt.subplot(3, 4, 3)
    plt.plot(history['transformer_loss'], label='Train Loss', color='blue')
    plt.plot(history['transformer_val_loss'], label='Val Loss', color='red')
    plt.title('Transformer Classifier Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 4. Transformer Training Accuracy
    plt.subplot(3, 4, 4)
    plt.plot(history['transformer_accuracy'], label='Train Acc', color='green')
    plt.plot(history['transformer_val_accuracy'], label='Val Acc', color='orange')
    plt.title('Transformer Classifier Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 5. Fraud Score Distribution - Normal Transactions
    plt.subplot(3, 4, 5)
    normal_scores = fraud_scores[test_y == 0]
    plt.hist(normal_scores, bins=50, alpha=0.7, color='green', label=f'Normal (n={len(normal_scores)})')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Fraud Scores: Normal Transactions')
    plt.xlabel('Fraud Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # 6. Fraud Score Distribution - Fraudulent Transactions
    plt.subplot(3, 4, 6)
    fraud_scores_fraud = fraud_scores[test_y == 1]
    plt.hist(fraud_scores_fraud, bins=50, alpha=0.7, color='red', label=f'Fraud (n={len(fraud_scores_fraud)})')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Fraud Scores: Fraudulent Transactions')
    plt.xlabel('Fraud Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # 7. Combined Score Distribution
    plt.subplot(3, 4, 7)
    plt.hist(normal_scores, bins=50, alpha=0.6, color='green', label='Normal')
    plt.hist(fraud_scores_fraud, bins=50, alpha=0.6, color='red', label='Fraud')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Combined Fraud Score Distribution')
    plt.xlabel('Fraud Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # 8. ROC Curve
    plt.subplot(3, 4, 8)
    fpr, tpr, _ = roc_curve(test_y, fraud_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - BankSafeNet')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # 9. Precision-Recall Curve
    plt.subplot(3, 4, 9)
    precision, recall, _ = precision_recall_curve(test_y, fraud_scores)
    avg_precision = auc(recall, precision)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # 10. Confusion Matrix
    plt.subplot(3, 4, 10)
    y_pred = [1 if score > threshold else 0 for score in fraud_scores]
    cm = confusion_matrix(test_y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    # 11. Model Architecture Summary (Text)
    plt.subplot(3, 4, 11)
    plt.text(0.1, 0.8, 'BankSafeNet Architecture:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, '• Autoencoder 1: Normal patterns', fontsize=10)
    plt.text(0.1, 0.6, '• Autoencoder 2: Anomaly detection', fontsize=10)
    plt.text(0.1, 0.5, '• Transformer: Sequential analysis', fontsize=10)
    plt.text(0.1, 0.4, '• Combined scoring system', fontsize=10)
    plt.text(0.1, 0.2, f'Total Parameters:', fontsize=10, fontweight='bold')
    ae1_params = model_dict['autoencoder1'].count_params()
    ae2_params = model_dict['autoencoder2'].count_params()
    tr_params = model_dict['transformer'].count_params()
    plt.text(0.1, 0.1, f'AE1: {ae1_params:,} | AE2: {ae2_params:,} | TR: {tr_params:,}', fontsize=9)
    plt.axis('off')
    
    # 12. Performance Metrics Summary
    plt.subplot(3, 4, 12)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics_text = f"""
    BANKSAFENET PERFORMANCE
    
    Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)
    Precision:    {precision_score:.4f} ({precision_score*100:.2f}%)
    Recall:       {recall_score:.4f} ({recall_score*100:.2f}%)
    F1 Score:     {f1:.4f} ({f1*100:.2f}%)
    FPR:          {fpr_rate:.4f}
    FNR:          {fnr_rate:.4f}
    """
    plt.text(0.1, 0.1, metrics_text, fontsize=10)
    plt.axis('off')
    
    # Finalize layout and save the plot to a PNG file instead of displaying it
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "evaluation.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation plot saved to {save_path}")