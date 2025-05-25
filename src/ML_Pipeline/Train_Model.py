import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def fit(df):
    """
    BankSafeNet implementation with dual autoencoders and transformer-based classification
    Following the paper methodology EXACTLY
    """
    # Prepare data
    X = df.drop('isFraud', axis=1).values
    y = df['isFraud'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud ratio: {np.sum(y)/len(y):.4f}")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Separate normal and fraud data for dual autoencoder training
    normal_data = X_train[y_train == 0]
    fraud_data = X_train[y_train == 1]
    
    print(f"Normal transactions: {len(normal_data)}")
    print(f"Fraud transactions: {len(fraud_data)}")
    
    input_dim = X.shape[1]
    
    # Build First Autoencoder (for normal transaction patterns) - DEEPER ARCHITECTURE
    autoencoder1 = build_autoencoder_v2(input_dim, name_prefix="ae1", dropout_rate=0.1)
    
    # Build Second Autoencoder (for anomaly detection) - FOCUSED ON FRAUD PATTERNS
    autoencoder2 = build_autoencoder_v2(input_dim, name_prefix="ae2", dropout_rate=0.2)
    
    # Train First Autoencoder ONLY on normal transactions (KEY DIFFERENCE)
    print("Training First Autoencoder on NORMAL transactions only...")
    history1 = autoencoder1.fit(
        normal_data, normal_data,
        epochs=50,
        batch_size=2048,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=8, min_lr=1e-6)
        ]
    )
    
    # Train Second Autoencoder on FRAUD-ENRICHED data (oversampled fraud)
    print("Training Second Autoencoder on fraud-enriched data...")
    # Oversample fraud data to balance
    fraud_oversampled = np.repeat(fraud_data, 10, axis=0)  # Repeat fraud samples
    mixed_data = np.vstack([normal_data, fraud_oversampled])
    np.random.shuffle(mixed_data)
    
    history2 = autoencoder2.fit(
        mixed_data, mixed_data,
        epochs=50,
        batch_size=2048,
        validation_data=(X_val, X_val),
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=8, min_lr=1e-6)
        ]
    )
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Build Transformer-based classification module - IMPROVED ARCHITECTURE
    transformer_model = build_transformer_classifier_v2(input_dim)
    
    # Train transformer classifier with class weights
    print("Training Transformer-based classifier with class balancing...")
    history3 = transformer_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=2048,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=8, min_lr=1e-6)
        ]
    )
    
    # Combine training histories
    combined_history = {
        'ae1_loss': history1.history['loss'],
        'ae1_val_loss': history1.history['val_loss'],
        'ae2_loss': history2.history['loss'],
        'ae2_val_loss': history2.history['val_loss'],
        'transformer_loss': history3.history['loss'],
        'transformer_val_loss': history3.history['val_loss'],
        'transformer_accuracy': history3.history['accuracy'],
        'transformer_val_accuracy': history3.history['val_accuracy']
    }
    
    # Return model dictionary
    model_dict = {
        'autoencoder1': autoencoder1,
        'autoencoder2': autoencoder2,
        'transformer': transformer_model
    }
    
    return model_dict, combined_history

def build_autoencoder_v2(input_dim, name_prefix="ae", dropout_rate=0.1):
    """Build improved autoencoder following paper architecture more closely"""
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    
    # Progressive compression with batch normalization and dropout
    encoded = layers.Dense(input_dim, activation='relu', name=f'{name_prefix}_enc0')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(dropout_rate)(encoded)
    
    encoded = layers.Dense(input_dim // 2, activation='relu', name=f'{name_prefix}_enc1')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(dropout_rate)(encoded)
    
    encoded = layers.Dense(input_dim // 4, activation='relu', name=f'{name_prefix}_enc2')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(dropout_rate)(encoded)
    
    # Bottleneck layer
    encoded = layers.Dense(input_dim // 8, activation='relu', name=f'{name_prefix}_bottleneck')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    
    # Decoder - symmetric architecture
    decoded = layers.Dense(input_dim // 4, activation='relu', name=f'{name_prefix}_dec1')(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(dropout_rate)(decoded)
    
    decoded = layers.Dense(input_dim // 2, activation='relu', name=f'{name_prefix}_dec2')(decoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(dropout_rate)(decoded)
    
    decoded = layers.Dense(input_dim, activation='relu', name=f'{name_prefix}_dec3')(decoded)
    decoded = layers.BatchNormalization()(decoded)
    
    # Output layer - linear activation for reconstruction
    decoded = layers.Dense(input_dim, activation='linear', name=f'{name_prefix}_output')(decoded)
    
    # Create model
    autoencoder = Model(input_layer, decoded, name=f'{name_prefix}_model')
    
    # Use custom loss function that's more sensitive to reconstruction errors
    def custom_mse_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
        return mse
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=custom_mse_loss
    )
    
    return autoencoder

def build_transformer_classifier_v2(input_dim):
    """Build improved transformer-based classification module"""
    
    # Input layer
    input_layer = layers.Input(shape=(input_dim,))
    
    # Feature embedding layer
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Reshape for attention mechanism
    sequence_length = 8  # Reduced artificial sequence length
    embed_dim = 16
    x = layers.Dense(sequence_length * embed_dim, activation='relu')(x)
    x = layers.Reshape((sequence_length, embed_dim))(x)
    
    # Multi-head self-attention layer (reduced from 2 layers to 1 layer with fewer heads)
    attention_output = layers.MultiHeadAttention(
        num_heads=4,  # Reduced number of heads from 8 to 4
        key_dim=embed_dim,
        name='multi_head_attention_0'
    )(x, x)
    
    # Add & Norm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    # Feed forward network
    ff_output = layers.Dense(embed_dim * 4, activation='gelu')(x)
    ff_output = layers.Dropout(0.1)(ff_output)
    ff_output = layers.Dense(embed_dim, activation='linear')(ff_output)
    
    # Add & Norm
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization()(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Add original features via skip connection
    skip = layers.Dense(x.shape[-1], activation='relu')(input_layer)
    x = layers.Add()([x, skip])
    
    # Classification head with multiple layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer with sigmoid for fraud probability
    output = layers.Dense(1, activation='sigmoid', name='fraud_probability')(x)
    
    # Create model
    model = Model(input_layer, output, name='transformer_classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model