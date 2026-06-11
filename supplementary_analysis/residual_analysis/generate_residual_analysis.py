from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "supplementary_analysis" / "residual_analysis" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD MODEL, TEST FEATURES AND TEST TARGET
# ============================================================

model = joblib.load(MODELS_DIR / "augmented_lightgbm_model.pkl")
X_test = pd.read_pickle(MODELS_DIR / "X_test_aug.pkl")
y_test_log = pd.read_pickle(MODELS_DIR / "y_test_aug.pkl")


# ============================================================
# 3. PREDICTIONS AND RESIDUALS IN €/m²
# ============================================================

# Model prediction is in log1p(price_m2)
y_pred_log = model.predict(X_test)

# Convert actual and predicted values back to €/m²
actual_eur = np.expm1(np.asarray(y_test_log).ravel())
predicted_eur = np.expm1(y_pred_log)

# Residual = observed - predicted
residuals_eur = actual_eur - predicted_eur


# ============================================================
# 4. BASIC METRICS
# ============================================================

r2 = r2_score(actual_eur, predicted_eur)
mae = mean_absolute_error(actual_eur, predicted_eur)
rmse = np.sqrt(mean_squared_error(actual_eur, predicted_eur))

mean_residual = np.mean(residuals_eur)
std_residual = np.std(residuals_eur)


# ============================================================
# 5. CLASSIC RESIDUAL PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    predicted_eur,
    residuals_eur,
    alpha=0.75,
    s=55,
    edgecolor="white",
    linewidth=0.6
)

# Zero residual reference line
plt.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Zero residual"
)

# Mean residual line
plt.axhline(
    mean_residual,
    color="red",
    linestyle="-",
    linewidth=2,
    label=f"Mean residual = {mean_residual:.2f} €/m²"
)

# Optional ±1 standard deviation bands
# plt.axhline(
#     mean_residual + std_residual,
#     color="grey",
#     linestyle=":",
#     linewidth=1.5,
#     label="+/- 1 standard deviation"
# )

plt.axhline(
    mean_residual - std_residual,
    color="grey",
    linestyle=":",
    linewidth=1.5
)

plt.xlabel("Predicted Advertised Rent per m² (€/m²)", fontsize=12)
plt.ylabel("Residuals (Actual - Predicted, €/m²)", fontsize=12)
plt.title("Residuals vs Predicted Values - Augmented LightGBM Model", fontsize=15, fontweight="bold")

plt.text(
    0.03,
    0.97,
    f"R² = {r2:.3f}\nMAE = {mae:.2f} €/m²\nRMSE = {rmse:.2f} €/m²\nn = {len(residuals_eur)}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="black",
        alpha=0.85
    )
)

plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()

output_path = OUTPUT_DIR / "residual_plot_predicted_vs_residuals_eur.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Residual plot saved to: {output_path}")


# ============================================================
# 6. SAVE RESIDUAL VALUES
# ============================================================

df_residuals = pd.DataFrame(
    {
        "actual_eur_m2": actual_eur,
        "predicted_eur_m2": predicted_eur,
        "residual_eur_m2": residuals_eur,
    },
    index=X_test.index
)

df_residuals.to_csv(OUTPUT_DIR / "residual_plot_values.csv", index=True)

print("Residual values saved to residual_plot_values.csv")