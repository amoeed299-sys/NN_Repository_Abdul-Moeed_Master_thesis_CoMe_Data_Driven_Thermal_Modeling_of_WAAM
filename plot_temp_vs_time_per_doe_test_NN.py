import numpy as np
import polars as pl
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------
# Same split as training
# -------------------------
def split_random(X, y, test_fraction=0.1, val_fraction=0.2, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=random_state, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------
# Paths (edit if needed)
# -------------------------
DATA_DIR  = Path("data/pre_processed")
MODEL_DIR = Path("trained_models")
OUT_DIR   = Path("evaluation_plots_temp_time_test_perDOE")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# DOE list (25)
# -------------------------
v_robot_list = [200, 400, 600, 800, 1000]
wire_feed_rate = [2, 4, 6, 8, 10]
doe_str = [f"{v_r}_{v_w}" for v_r in v_robot_list for v_w in wire_feed_rate]

# -------------------------
# Loop DOE and save 25 plots
# -------------------------
for exp in doe_str:
    parquet_path = DATA_DIR / f"{exp}.parquet"
    model_path   = MODEL_DIR / f"model_{exp}.h5"

    if not parquet_path.exists():
        print(f"[SKIP] Missing parquet: {parquet_path}")
        continue
    if not model_path.exists():
        print(f"[SKIP] Missing model: {model_path}")
        continue

    print(f"[INFO] Plotting {exp}")

    # Load data
    df = pl.read_parquet(parquet_path).select(["time", "x", "y", "z", "Temp_K"])
    X = df.select(["time", "x", "y", "z"]).to_numpy()
    y = df.select(["Temp_K"]).to_numpy()

    # Same split as training
    X_train, X_val, X_test, y_train, y_val, y_test = split_random(
        X, y, test_fraction=0.1, val_fraction=0.2, random_state=42
    )

    # Rebuild scalers (fit on train only)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    # Load model and predict on TEST
    model = tf.keras.models.load_model(model_path, compile=False)
    X_test_scaled = scaler_X.transform(X_test)
    y_pred_test_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled).flatten()

    # Mean temp vs time (TEST)
    df_test = pd.DataFrame({
        "time": X_test[:, 0],
        "T_true": y_test.flatten(),
        "T_pred": y_pred_test
    }).groupby("time", as_index=False).mean(numeric_only=True).sort_values("time")

    # Plot and save
    plt.figure(figsize=(10, 5))
    plt.plot(df_test["time"], df_test["T_true"], label="Original")
    plt.plot(df_test["time"], df_test["T_pred"], linestyle="--", label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Temperature (K)")
    plt.title(f"Temperature vs Time — Original vs Predicted ({exp})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"temp_vs_time_{exp}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"[DONE] Saved plots to: {OUT_DIR.resolve()}")
