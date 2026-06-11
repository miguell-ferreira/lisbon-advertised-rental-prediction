from pathlib import Path
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "supplementary_analysis" / "ice_plots" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "augmented_lightgbm_model.pkl"
X_TEST_PATH = MODELS_DIR / "X_test_aug.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not X_TEST_PATH.exists():
    raise FileNotFoundError(f"X_test file not found: {X_TEST_PATH}")

lgbm_final = joblib.load(MODEL_PATH)
X_total = pd.read_pickle(X_TEST_PATH)


features_to_plot = [
    "Densidade_AL_km2",
    "Taxa_Conversao_Reabilitacao",
    "Stock_12M_ALV_Reabilitacao_Count",
    "Densidade_Museus_km2",
    "Pct_Vagos_Outros_Motivos_2021",

    "MeanNightimeRadianceVIIRS",
    "Proporcao_Populacao_Residente_Estrangeira_2021",
    "PCT_Edificios_Antigos",
    "PCT_Edificios_Modernos",
    "Densidade_Populacional_2021",
    "Densidade_Alojamentos_Familiares_Classicos_2021",
    "Indice_Pressao_Habitacional_2021",
    "Indice_Envelhecimento_2021",
    "Percentagem_Residencia_Secundaria_2021",
    "Liquidez_Vagos_2021",
    "Delta_Crescimento_Vagos_Outros_Motivos_11_21",
]

feature_titles = {
    "Densidade_AL_km2": "ICE Plot - Local Accommodation Density",
    "Taxa_Conversao_Reabilitacao": "ICE Plot - Conversion Rate of Rehabilitation Projects",
    "Stock_12M_ALV_Reabilitacao_Count": "ICE Plot - Recent Stock of Approved Rehabilitation Projects",
    "Densidade_Museus_km2": "ICE Plot - Museum Density per km²",
    "Pct_Vagos_Outros_Motivos_2021": "ICE Plot - Housing Vacancy for Other Motives",

    "MeanNightimeRadianceVIIRS": "ICE Plot - Nighttime Light Intensity",
    "Proporcao_Populacao_Residente_Estrangeira_2021": "ICE Plot - Foreign Resident Population Share",
    "PCT_Edificios_Antigos": "ICE Plot - Share of Old Buildings",
    "PCT_Edificios_Modernos": "ICE Plot - Share of Modern Buildings",
    "Densidade_Populacional_2021": "ICE Plot - Population Density",
    "Densidade_Alojamentos_Familiares_Classicos_2021": "ICE Plot - Conventional Dwelling Density",
    "Indice_Pressao_Habitacional_2021": "ICE Plot - Housing Pressure Index",
    "Indice_Envelhecimento_2021": "ICE Plot - Ageing Index",
    "Percentagem_Residencia_Secundaria_2021": "ICE Plot - Secondary Residence Share",
    "Liquidez_Vagos_2021": "ICE Plot - Vacant Housing Market Liquidity",
    "Delta_Crescimento_Vagos_Outros_Motivos_11_21": "ICE Plot - Growth of Vacancy for Other Motives",
}

def safe_filename(name: str) -> str:
    """
    Converts feature names into safe file names.
    """
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return name[:120]


def clean_axis_label(feature_name: str) -> str:
    """
    Converts variable names into readable axis labels.
    """
    return feature_name.replace("_", " ")

available_features = [feature for feature in features_to_plot if feature in X_total.columns]
missing_features = [feature for feature in features_to_plot if feature not in X_total.columns]

print("\nGenerating ICE plots for:")
for feature in available_features:
    print(f" - {feature}")

if missing_features:
    print("\nVariables not found in X_test_aug and therefore skipped:")
    for feature in missing_features:
        print(f" - {feature}")

if not available_features:
    raise ValueError("No valid features found. Please check feature names in X_test_aug.pkl.")

for feature in available_features:

    fig, ax = plt.subplots(figsize=(8, 6))

    PartialDependenceDisplay.from_estimator(
        lgbm_final,
        X_total,
        features=[feature],
        kind="both",
        subsample=100,
        random_state=42,
        grid_resolution=50,
        ice_lines_kw={
            "color": "tab:blue",
            "alpha": 0.2,
            "linewidth": 0.5,
        },
        pd_line_kw={
            "color": "tab:red",
            "linewidth": 4,
        },
        ax=ax,
    )

    ax.set_title(
        feature_titles.get(feature, f"ICE Plot - {feature}"),
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel(clean_axis_label(feature))
    ax.set_ylabel("Predicted Log Rental Price per m²")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"ice_{safe_filename(feature)}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


print(f"\nAll ICE plots saved to: {OUTPUT_DIR}")