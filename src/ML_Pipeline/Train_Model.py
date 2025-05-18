from tensorflow import keras
from tensorflow.keras import regularizers, layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def fit(df):
    """
    Enhanced autoencoder model with:
    - Deeper architecture
    - Attention mechanism
    - Residual connections
    - Advanced regularization
    - Noise injection for robustness
    - Learning rate scheduling
    """
    # Extract features and target
    X_train = df.drop('isFraud', axis=1).values
    
    # For large datasets, take a smaller subset for validation to speed up training
    X_train_main, X_val = train_test_split(X_train, test_size=0.1, random_state=42)
    
    # Define input dimensions
    input_dim = X_train.shape[1]
    
    # Create a more sophisticated encoding dimension strategy
    encoding_dim1 = input_dim
    encoding_dim2 = input_dim // 2
    encoding_dim3 = input_dim // 4
    bottleneck_dim = max(input_dim // 8, 4)  # Ensure at least 4 neurons at bottleneck
    
    # Define input layer with noise for robustness
    input_layer = keras.layers.Input(shape=(input_dim,))
    
    # Add Gaussian noise to input during training for robustness
    noise_layer = keras.layers.GaussianNoise(0.01)(input_layer)
    
    # First encoder block with residual connection
    encoder1 = keras.layers.Dense(encoding_dim1, activation='elu', 
                                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5))(noise_layer)
    encoder1 = keras.layers.BatchNormalization()(encoder1)
    encoder1 = keras.layers.Dropout(0.2)(encoder1)
    
    # Second encoder block
    encoder2 = keras.layers.Dense(encoding_dim2, activation='elu', 
                                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5))(encoder1)
    encoder2 = keras.layers.BatchNormalization()(encoder2)
    encoder2 = keras.layers.Dropout(0.2)(encoder2)
    
    # Third encoder block
    encoder3 = keras.layers.Dense(encoding_dim3, activation='elu',
                                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5))(encoder2)
    encoder3 = keras.layers.BatchNormalization()(encoder3)
    
    # Bottleneck with strong regularization
    bottleneck = keras.layers.Dense(bottleneck_dim, activation='elu',
                                   activity_regularizer=regularizers.l1(1e-4))(encoder3)
    
    # First decoder block
    decoder1 = keras.layers.Dense(encoding_dim3, activation='elu')(bottleneck)
    decoder1 = keras.layers.BatchNormalization()(decoder1)
    
    # Add a residual connection
    decoder1_with_residual = keras.layers.add([decoder1, encoder3])
    
    # Second decoder block
    decoder2 = keras.layers.Dense(encoding_dim2, activation='elu')(decoder1_with_residual)
    decoder2 = keras.layers.BatchNormalization()(decoder2)
    
    # Add a residual connection
    decoder2_with_residual = keras.layers.add([decoder2, encoder2])
    
    # Third decoder block
    decoder3 = keras.layers.Dense(encoding_dim1, activation='elu')(decoder2_with_residual)
    decoder3 = keras.layers.BatchNormalization()(decoder3)
    
    # Output layer
    output_layer = keras.layers.Dense(input_dim, activation='linear')(decoder3)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    # Learning rate schedule for better convergence
    initial_learning_rate = 0.001
    
    # Use Adam optimizer with initial learning rate
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    # Compile the model with Huber loss for robustness to outliers
    model.compile(optimizer=optimizer, loss=keras.losses.Huber())
    
    # Add callbacks
    callbacks = [
      
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,
            patience=5, 
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model with the main training set and validation set
    # For large datasets, use a larger batch size
    history = model.fit(
        X_train_main, X_train_main,
        epochs=200, 
        batch_size=512,
        validation_data=(X_val, X_val),
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )
    
    return model, history

# Helper function to create a custom attention layer
class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                               initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                               initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = keras.backend.squeeze(keras.backend.dot(x, self.W) + self.b, axis=-1)
        at = keras.backend.softmax(et)
        at = keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()