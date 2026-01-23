import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D,
    BatchNormalization, ReLU,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
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

def build_baseline_cnn(input_shape):
    model = Sequential()

    # Block 1
    model.add(Conv1D(32, kernel_size=5, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))

    # Block 2
    model.add(Conv1D(64, kernel_size=5, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))

    # Block 3
    model.add(Conv1D(128, kernel_size=5, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=2))

    # Classifier
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))

    return model

def main():
    # --- Data Loading (Step 1) ---
    print("--- Loading Data ---")
    train_path = "./data/mitbih_train.csv"
    test_path = "./data/mitbih_test.csv"
    
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train = train_df.iloc[:, :-1].values
    y_train = relabel(train_df.iloc[:, -1].values)
    
    X_test = test_df.iloc[:, :-1].values
    y_test = relabel(test_df.iloc[:, -1].values)

    # --- Preprocessing (Step 2) ---
    print("--- Normalizing Data (Sample-wise) ---")
    X_train_norm = normalize_per_sample(X_train)
    X_test_norm = normalize_per_sample(X_test)

    X_train_cnn = X_train_norm[..., np.newaxis]
    X_test_cnn = X_test_norm[..., np.newaxis]

    print("--- Computing Class Weights ---")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class Weights: {class_weight_dict}")

    # --- Model Building (Step 3) ---
    print("\n--- Building Baseline CNN ---")
    model = build_baseline_cnn(input_shape=(187, 1))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision")
        ]
    )

    model.summary()

    # --- Training ---
    print("\n--- Training Model ---")
    history = model.fit(
        X_train_cnn,
        y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=256,
        class_weight=class_weight_dict,
        verbose=1
    )

    # --- Evaluation ---
    print("\n--- Evaluating on Test Set ---")
    test_results = model.evaluate(X_test_cnn, y_test, verbose=0)

    print(f"\nTest Accuracy  : {test_results[1]:.4f}")
    print(f"Test Recall    : {test_results[2]:.4f}")
    print(f"Test Precision : {test_results[3]:.4f}")

    # Log metrics to check for overfitting
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
