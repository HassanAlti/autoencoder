import numpy as np

def init(test_df, model_dict, columns):
    """
    BankSafeNet fraud prediction - CORRECTED to match paper methodology
    """
    # Ensure correct column order
    if 'isFraud' in test_df.columns:
        X = test_df.drop('isFraud', axis=1)[columns].values
    else:
        X = test_df[columns].values
    
    # Extract models
    autoencoder1 = model_dict['autoencoder1']  # Trained ONLY on normal patterns
    autoencoder2 = model_dict['autoencoder2']  # Trained on mixed/fraud-enriched data
    transformer = model_dict['transformer']    # Sequential relationship classifier
    
    # Get reconstructions
    recon1 = autoencoder1.predict(X, verbose=0)
    recon2 = autoencoder2.predict(X, verbose=0)
    
    # Calculate reconstruction errors (MSE) as per paper
    mse1 = np.mean(np.square(recon1 - X), axis=1)  # Error from normal-trained AE
    mse2 = np.mean(np.square(recon2 - X), axis=1)  # Error from mixed-trained AE
    
    # Get transformer classification probability
    transformer_prob = transformer.predict(X, verbose=0).flatten()
    
    # CORRECTED SCORING METHODOLOGY based on paper logic:
    # 1. Normal transactions should reconstruct well with AE1 (low mse1)
    # 2. Fraud transactions should reconstruct poorly with AE1 (high mse1)  
    # 3. AE2 trained on mixed data should have different reconstruction patterns
    # 4. Transformer provides direct fraud probability
    
    # Normalize scores to [0, 1] range
    mse1_norm = robust_normalize(mse1)
    mse2_norm = robust_normalize(mse2)
    
    # Key insight: Use reconstruction error RATIO instead of just individual errors
    reconstruction_ratio = np.where(mse2_norm > 0, mse1_norm / (mse2_norm + 1e-8), mse1_norm)
    reconstruction_ratio = robust_normalize(reconstruction_ratio)
    
    # Combine scores with CORRECTED weights based on paper methodology
    # Higher weight to transformer (more reliable), reconstruction ratio as anomaly indicator
    combined_scores = (
        0.6 * transformer_prob +           # Primary fraud indicator
        0.3 * reconstruction_ratio +       # Anomaly detection from dual AE
        0.1 * mse1_norm                   # Raw anomaly score from normal AE
    )
    
    return combined_scores

def robust_normalize(scores):
    """Robust normalization using percentiles to handle outliers"""
    p5, p95 = np.percentile(scores, [5, 95])
    scores_clipped = np.clip(scores, p5, p95)
    
    min_score = np.min(scores_clipped)
    max_score = np.max(scores_clipped)
    
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    
    normalized = (scores_clipped - min_score) / (max_score - min_score)
    return normalized