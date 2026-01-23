import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import os

def relabel(y):
    return np.where(y == 0, 0, 1)

def normalize_per_sample(X):
    """
    Normalizes each ECG beat independently.
    Computationally equivalent to: (row - row_mean) / row_std
    """
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        row = X[i, :].reshape(-1, 1)
        scaler = StandardScaler()
        X_norm[i, :] = scaler.fit_transform(row).ravel()
    return X_norm

def main():
    # --- Step 1 Recap (Loading) ---
    print("--- Loading Data (Step 1 Recap) ---")
    train_path = "./data/mitbih_train.csv"
    test_path = "./data/mitbih_test.csv"

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train = train_df.iloc[:, :-1].values
    y_train_raw = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test_raw = test_df.iloc[:, -1].values

    y_train = relabel(y_train_raw)
    y_test = relabel(y_test_raw)

    # --- Step 2.1: Normalize ECG signals (sample-wise) ---
    print("\n--- Step 2.1: Normalizing ECG signals (sample-wise) ---")
    print("This may take a moment...")
    X_train_norm = normalize_per_sample(X_train)
    X_test_norm = normalize_per_sample(X_test)
    print("Normalization complete.")

    # --- Step 2.2: Reshape for 1D CNN ---
    print("\n--- Step 2.2: Reshaping for 1D CNN ---")
    X_train_cnn = X_train_norm[..., np.newaxis]
    X_test_cnn = X_test_norm[..., np.newaxis]

    print(f"X_train_cnn shape: {X_train_cnn.shape}")
    print(f"X_test_cnn shape: {X_test_cnn.shape}")

    # --- Step 2.3: Handle class imbalance (Class Weights) ---
    print("\n--- Step 2.3: Computing Class Weights ---")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )

    class_weight_dict = {
        0: class_weights[0],
        1: class_weights[1]
    }
    print(f"Class Weight Dictionary: {class_weight_dict}")
    print(f"Interpretation: Abnormal class (1) is weighted {class_weight_dict[1]/class_weight_dict[0]:.2f}x more than Normal (0)")

    # --- Step 2.4: Final sanity checks ---
    print("\n--- Step 2.4: Final Sanity Checks ---")
    print(f"Train - Mean: {X_train_cnn.mean():.6f}, Std: {X_train_cnn.std():.6f}")
    print(f"Test  - Mean: {X_test_cnn.mean():.6f}, Std: {X_test_cnn.std():.6f}")

    if np.isclose(X_train_cnn.mean(), 0, atol=1e-2) and np.isclose(X_train_cnn.std(), 1, atol=1e-2):
        print("\n✅ Step 2 Accomplished Successfully: Data is normalized and reshaped.")
    else:
        print("\n⚠️ Warning: Normalization sanity check failed. Check your data.")

if __name__ == "__main__":
    main()
