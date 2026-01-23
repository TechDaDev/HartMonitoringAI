import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU,
    Add, MaxPooling1D, Flatten, Dense, Dropout
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def relabel(y):
    return np.where(y == 0, 0, 1)

def normalize_per_sample(X):
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        row = X[i, :].reshape(-1, 1)
        scaler = StandardScaler()
        X_norm[i, :] = scaler.fit_transform(row).ravel()
    return X_norm

def residual_block(x, filters, kernel_size=5, pool_size=5):
    shortcut = x

    # First Conv Layer
    x = Conv1D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second Conv Layer
    x = Conv1D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)

    # Match dimensions for residual connection if necessary
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same")(shortcut)

    # Add skip connection
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    # Pooling
    x = MaxPooling1D(pool_size=pool_size, strides=2, padding="same")(x)

    return x

def build_residual_cnn(input_shape):
    inputs = Input(shape=input_shape)

    # Initial Block
    x = Conv1D(32, 5, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Five residual blocks (as per paper)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    # Classifier Head
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model

def main():
    # --- Data Loading & Preprocessing ---
    print("--- Loading Data ---")
    train_path = "./data/mitbih_train.csv"
    test_path = "./data/mitbih_test.csv"
    
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Shuffle training data to ensure validation split is representative
    train_df = shuffle(train_df, random_state=42)

    X_train = train_df.iloc[:, :-1].values
    y_train = relabel(train_df.iloc[:, -1].values)
    
    X_test = test_df.iloc[:, :-1].values
    y_test = relabel(test_df.iloc[:, -1].values)

    print("--- Normalizing Data (Sample-wise) ---")
    X_train_cnn = normalize_per_sample(X_train)[..., np.newaxis]
    X_test_cnn = normalize_per_sample(X_test)[..., np.newaxis]

    print("--- Computing Class Weights ---")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class Weights: {class_weight_dict}")

    # --- Build Residual Model ---
    print("\n--- Building Deep Residual CNN ---")
    res_model = build_residual_cnn(input_shape=(187, 1))

    res_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Lower LR for stability
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision")
        ]
    )

    res_model.summary()

    # --- Training ---
    print("\n--- Training Deep Residual Model (25 Epochs) ---")
    history_res = res_model.fit(
        X_train_cnn,
        y_train,
        validation_split=0.1,
        epochs=25,
        batch_size=256,
        class_weight=class_weight_dict,
        verbose=1
    )

    # --- Evaluation ---
    print("\n--- Evaluating on Test Set ---")
    test_results = res_model.evaluate(X_test_cnn, y_test, verbose=0)

    print(f"\nTest Accuracy  : {test_results[1]:.4f}")
    print(f"Test Recall    : {test_results[2]:.4f}")
    print(f"Test Precision : {test_results[3]:.4f}")

    # Overfitting Check
    train_acc = history_res.history['accuracy'][-1]
    val_acc = history_res.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")

    # Save the model
    res_model.save("models/mitbih_residual_cnn.h5")
    print("\nModel saved as models/mitbih_residual_cnn.h5")

if __name__ == "__main__":
    main()
