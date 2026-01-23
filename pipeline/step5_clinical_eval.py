import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import os

# Set matplotlib to non-interactive mode
plt.switch_backend('Agg')

def relabel(y):
    return np.where(y == 0, 0, 1)

def normalize_per_sample(X):
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        row = X[i, :].reshape(-1, 1)
        scaler = StandardScaler()
        X_norm[i, :] = scaler.fit_transform(row).ravel()
    return X_norm

def main():
    # --- Load Test Data ---
    print("--- Loading and Preprocessing Test Data ---")
    test_path = "./data/mitbih_test.csv"
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        return

    test_df = pd.read_csv(test_path, header=None)
    X_test = test_df.iloc[:, :-1].values
    y_test = relabel(test_df.iloc[:, -1].values)
    X_test_cnn = normalize_per_sample(X_test)[..., np.newaxis]

    # --- Load Model ---
    model_path = "models/mitbih_residual_cnn.h5"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run Step 4 first.")
        return
    
    print(f"--- Loading Model: {model_path} ---")
    res_model = tf.keras.models.load_model(model_path)

    # --- 5.1 Confusion Matrix ---
    print("\n--- Step 5.1: Generating Confusion Matrix ---")
    y_probs = res_model.predict(X_test_cnn)
    y_pred = (y_probs > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Abnormal"]
    )
    disp.plot(cmap="Blues")
    plt.title("ECG Beat Classification Confusion Matrix (Threshold=0.5)")
    plt.savefig("plots/confusion_matrix_05.png")
    print("Confusion matrix plot saved as plots/confusion_matrix_05.png")

    # --- 5.2 Sensitivity & Specificity ---
    print("\n--- Step 5.2: Clinical Metrics (Threshold=0.5) ---")
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)   # Recall for Abnormal
    specificity = TN / (TN + FP)   # True Normal detection

    print(f"Sensitivity (Abnormal detection): {sensitivity:.4f}")
    print(f"Specificity (Normal detection):   {specificity:.4f}")

    # --- 5.3 Alarm threshold tuning ---
    print("\n--- Step 5.3: Alarm Threshold Tuning ---")
    for t in [0.2, 0.3, 0.4, 0.5]:
        y_pred_t = (y_probs > t).astype(int)
        cm_t = confusion_matrix(y_test, y_pred_t)
        TN_t, FP_t, FN_t, TP_t = cm_t.ravel()

        sens = TP_t / (TP_t + FN_t)
        spec = TN_t / (TN_t + FP_t)

        print(f"Threshold {t:.1f}: Sensitivity={sens:.4f}, Specificity={spec:.4f}, Missed Alarms (FN)={FN_t}")

    print("\n--- Step 5.4: Clinical Statement ---")
    print("“The system prioritizes sensitivity over specificity to minimize missed abnormal cardiac events.")
    print("While this may increase false alarms, it aligns with clinical safety requirements for continuous monitoring systems.”")

if __name__ == "__main__":
    main()
