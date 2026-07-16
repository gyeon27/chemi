from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "output" / "delta_e_transition_dataset.csv"
PERIODIC_PATH = ROOT / "data" / "reference" / "periodic_descriptors.csv"
OUT_DIR = ROOT / "output" / "enriched_property_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)


BASIC_NUMERIC = [
    "Atomic Number",
    "Atomic Radius pm",
    "Atomic Mass",
    "Electronegativity",
    "First Ionization Energy kJ mol-1",
    "Density g cm-3",
    "Melting Point C",
    "Valence Electrons",
    "d Electrons",
]

DERIVED_NUMERIC = [
    "Period",
    "Group",
    "Molar Volume cm3 mol-1",
    "Radius Cubed pm3",
    "Mass per Radius Cubed",
    "Density per Atomic Mass",
    "d Electrons Squared",
    "Distance from d5",
    "Distance from d10",
    "Valence Electrons Squared",
    "Radius x EN",
    "EN x IE",
    "Melting Point x Density",
    "Is d Block",
    "Is s Block",
    "Is p Block",
]

CATEGORICAL = ["Block", "Metal Family"]

INNER_CV = KFold(n_splits=5, shuffle=True, random_state=42)


def make_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    periodic = pd.read_csv(PERIODIC_PATH)
    df = df.merge(periodic, on="Element", how="left")

    df["Molar Volume cm3 mol-1"] = df["Atomic Mass"] / df["Density g cm-3"]
    df["Radius Cubed pm3"] = df["Atomic Radius pm"] ** 3
    df["Mass per Radius Cubed"] = df["Atomic Mass"] / df["Radius Cubed pm3"]
    df["Density per Atomic Mass"] = df["Density g cm-3"] / df["Atomic Mass"]
    df["d Electrons Squared"] = df["d Electrons"] ** 2
    df["Distance from d5"] = (df["d Electrons"] - 5).abs()
    df["Distance from d10"] = (df["d Electrons"] - 10).abs()
    df["Valence Electrons Squared"] = df["Valence Electrons"] ** 2
    df["Radius x EN"] = df["Atomic Radius pm"] * df["Electronegativity"]
    df["EN x IE"] = df["Electronegativity"] * df["First Ionization Energy kJ mol-1"]
    df["Melting Point x Density"] = df["Melting Point C"] * df["Density g cm-3"]
    df["Is d Block"] = (df["Block"] == "d").astype(int)
    df["Is s Block"] = (df["Block"] == "s").astype(int)
    df["Is p Block"] = (df["Block"] == "p").astype(int)
    return df


def preprocessor(use_categorical: bool = True) -> ColumnTransformer:
    transformers = [("num", StandardScaler(), BASIC_NUMERIC + DERIVED_NUMERIC)]
    if use_categorical:
        transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL))
    return ColumnTransformer(transformers)


def models() -> dict[str, object]:
    alphas = np.logspace(-4, 3, 80)
    prep = preprocessor(use_categorical=True)
    return {
        "Linear": Pipeline([("prep", prep), ("model", LinearRegression())]),
        "Ridge": Pipeline(
            [
                ("prep", prep),
                ("model", RidgeCV(alphas=alphas, cv=INNER_CV, scoring="neg_mean_squared_error")),
            ]
        ),
        "LASSO": Pipeline(
            [
                ("prep", prep),
                ("model", LassoCV(alphas=alphas, cv=INNER_CV, max_iter=200000, random_state=42)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    ElasticNetCV(
                        alphas=alphas,
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        cv=INNER_CV,
                        max_iter=200000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Poly2 Ridge": Pipeline(
            [
                ("prep", preprocessor(use_categorical=False)),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale2", StandardScaler()),
                ("model", RidgeCV(alphas=alphas, cv=INNER_CV, scoring="neg_mean_squared_error")),
            ]
        ),
        "RandomForest comparison": Pipeline(
            [
                ("prep", prep),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_row(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
    }


def evaluate_energy_targets(df: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "BCC_minus_FCC": df["E_BCC"] - df["E_FCC"],
        "HCP_minus_FCC": df["E_HCP"] - df["E_FCC"],
        "BCC_minus_HCP": df["E_BCC"] - df["E_HCP"],
    }
    rows = []
    for target_name, y in targets.items():
        for model_name, model in models().items():
            pred = cross_val_predict(clone(model), x, y, cv=LeaveOneOut())
            rows.append({"Target": target_name, "Model": model_name, **metric_row(y, pred)})
    return pd.DataFrame(rows)


def evaluate_structure_prediction(df: pd.DataFrame, x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    predictions = []
    for model_name, model in models().items():
        pred_bf = cross_val_predict(clone(model), x, df["E_BCC"] - df["E_FCC"], cv=LeaveOneOut())
        pred_hf = cross_val_predict(clone(model), x, df["E_HCP"] - df["E_FCC"], cv=LeaveOneOut())
        for i, row in df.iterrows():
            energies = {"BCC": pred_bf[i], "FCC": 0.0, "HCP": pred_hf[i]}
            pred_structure = min(energies, key=energies.get)
            predictions.append(
                {
                    "Model": model_name,
                    "Element": row["Element"],
                    "Predicted Structure": pred_structure,
                    "OQMD Stable Structure": row["DFT Stable Structure"],
                    "Actual Structure": row["Actual Structure"],
                    "Pred_BCC_minus_FCC": pred_bf[i],
                    "Pred_HCP_minus_FCC": pred_hf[i],
                    "Matches OQMD": pred_structure == row["DFT Stable Structure"],
                    "Matches Actual": pred_structure == row["Actual Structure"],
                }
            )
        pred_df = pd.DataFrame([p for p in predictions if p["Model"] == model_name])
        summary.append(
            {
                "Model": model_name,
                "Accuracy vs OQMD": accuracy_score(pred_df["OQMD Stable Structure"], pred_df["Predicted Structure"]),
                "Accuracy vs Actual": accuracy_score(pred_df["Actual Structure"], pred_df["Predicted Structure"]),
                "Wrong vs OQMD": ", ".join(pred_df.loc[~pred_df["Matches OQMD"], "Element"]),
                "Wrong vs Actual": ", ".join(pred_df.loc[~pred_df["Matches Actual"], "Element"]),
            }
        )
    return pd.DataFrame(summary), pd.DataFrame(predictions)


def final_ridge_equations(df: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model = models()["Ridge"]
    targets = {
        "BCC_minus_FCC": df["E_BCC"] - df["E_FCC"],
        "HCP_minus_FCC": df["E_HCP"] - df["E_FCC"],
    }
    for target_name, y in targets.items():
        fitted = clone(model).fit(x, y)
        prep = fitted.named_steps["prep"]
        feature_names = prep.get_feature_names_out()
        ridge = fitted.named_steps["model"]
        rows.append({"Target": target_name, "Term": "Intercept", "Coefficient": ridge.intercept_})
        for name, coef in zip(feature_names, ridge.coef_):
            rows.append({"Target": target_name, "Term": name, "Coefficient": coef})
        rows.append({"Target": target_name, "Term": "selected alpha", "Coefficient": ridge.alpha_})
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Found unknown categories in columns",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    df = make_dataset()
    x = df[BASIC_NUMERIC + DERIVED_NUMERIC + CATEGORICAL]

    enriched = df[["Element"] + BASIC_NUMERIC + DERIVED_NUMERIC + CATEGORICAL]
    energy_metrics = evaluate_energy_targets(df, x)
    structure_summary, structure_predictions = evaluate_structure_prediction(df, x)
    equations = final_ridge_equations(df, x)

    enriched.to_csv(OUT_DIR / "enriched_features.csv", index=False, encoding="utf-8-sig")
    energy_metrics.to_csv(OUT_DIR / "loocv_energy_metrics.csv", index=False, encoding="utf-8-sig")
    structure_summary.to_csv(OUT_DIR / "loocv_structure_accuracy.csv", index=False, encoding="utf-8-sig")
    structure_predictions.to_csv(OUT_DIR / "loocv_structure_predictions.csv", index=False, encoding="utf-8-sig")
    equations.to_csv(OUT_DIR / "final_ridge_equations.csv", index=False, encoding="utf-8-sig")

    best = structure_summary.sort_values("Accuracy vs OQMD", ascending=False).iloc[0]
    best_pred = structure_predictions[structure_predictions["Model"] == best["Model"]]
    labels = ["BCC", "FCC", "HCP"]
    cm_oqmd = pd.DataFrame(
        confusion_matrix(best_pred["OQMD Stable Structure"], best_pred["Predicted Structure"], labels=labels),
        index=[f"OQMD {x}" for x in labels],
        columns=[f"Pred {x}" for x in labels],
    )
    cm_actual = pd.DataFrame(
        confusion_matrix(best_pred["Actual Structure"], best_pred["Predicted Structure"], labels=labels),
        index=[f"Actual {x}" for x in labels],
        columns=[f"Pred {x}" for x in labels],
    )
    cm_oqmd.to_csv(OUT_DIR / "best_confusion_matrix_vs_oqmd.csv", encoding="utf-8-sig")
    cm_actual.to_csv(OUT_DIR / "best_confusion_matrix_vs_actual.csv", encoding="utf-8-sig")

    print(f"Saved outputs to {OUT_DIR}")
    print("\nEnergy target LOOCV metrics:")
    print(energy_metrics.sort_values(["Target", "RMSE"]).to_string(index=False))
    print("\nStructure prediction LOOCV accuracy:")
    print(structure_summary.to_string(index=False))
    print(f"\nBest structure model: {best['Model']}")
    print("\nConfusion matrix vs OQMD:")
    print(cm_oqmd.to_string())
    print("\nConfusion matrix vs actual:")
    print(cm_actual.to_string())


if __name__ == "__main__":
    main()
