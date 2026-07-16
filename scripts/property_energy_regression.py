from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "output" / "delta_e_transition_dataset.csv"
OUT_DIR = ROOT / "output" / "property_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Atomic Radius pm",
    "Electronegativity",
    "First Ionization Energy kJ mol-1",
    "Density g cm-3",
    "Melting Point C",
    "d Electrons",
]

INNER_CV = KFold(n_splits=5, shuffle=True, random_state=42)

PAIR_TARGETS = {
    "BCC_minus_FCC": ("E_BCC", "E_FCC"),
    "HCP_minus_FCC": ("E_HCP", "E_FCC"),
    "BCC_minus_HCP": ("E_BCC", "E_HCP"),
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
    }


def vif_table(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in x.columns:
        y = x[col].to_numpy()
        others = x.drop(columns=[col]).to_numpy()
        pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
        y_pred = cross_val_predict(pipe, others, y, cv=LeaveOneOut())
        r2 = r2_score(y, y_pred)
        rows.append({"Feature": col, "LOOCV_R2_when_predicted_by_others": r2})
    # Classical VIF should be fitted in-sample. It is included for collinearity diagnosis.
    for row in rows:
        col = row["Feature"]
        y = x[col].to_numpy()
        others = x.drop(columns=[col]).to_numpy()
        pipe = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(others, y)
        r2_in = pipe.score(others, y)
        row["VIF"] = float(1 / (1 - r2_in)) if r2_in < 1 else np.inf
    return pd.DataFrame(rows)


def make_models() -> dict[str, object]:
    alphas = np.logspace(-4, 3, 80)
    return {
        "Multiple Linear Regression": Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=alphas, cv=INNER_CV, scoring="neg_mean_squared_error")),
            ]
        ),
        "LASSO": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LassoCV(alphas=alphas, cv=INNER_CV, max_iter=200000, random_state=42),
                ),
            ]
        ),
        "Elastic Net": Pipeline(
            [
                ("scale", StandardScaler()),
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
        "Polynomial Ridge degree 2": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=alphas, cv=INNER_CV, scoring="neg_mean_squared_error")),
            ]
        ),
        "Random Forest comparison": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            random_state=42,
        ),
    }


def target_cv_table(df: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = make_models()
    for target_name, (a, b) in PAIR_TARGETS.items():
        y = (df[a] - df[b]).to_numpy()
        for model_name, model in models.items():
            y_pred = cross_val_predict(clone(model), x, y, cv=LeaveOneOut())
            rows.append({"Target": target_name, "Model": model_name, **cv_metrics(y, y_pred)})
    return pd.DataFrame(rows)


def predict_structures_from_pair_models(
    df: pd.DataFrame,
    x: pd.DataFrame,
    base_model: object,
    cv,
) -> pd.DataFrame:
    pred_bcc_minus_fcc = cross_val_predict(
        clone(base_model), x, (df["E_BCC"] - df["E_FCC"]).to_numpy(), cv=cv
    )
    pred_hcp_minus_fcc = cross_val_predict(
        clone(base_model), x, (df["E_HCP"] - df["E_FCC"]).to_numpy(), cv=cv
    )

    rows = []
    for idx, row in df.iterrows():
        energies = {
            "BCC": pred_bcc_minus_fcc[idx],
            "FCC": 0.0,
            "HCP": pred_hcp_minus_fcc[idx],
        }
        pred_structure = min(energies, key=energies.get)
        rows.append(
            {
                "Element": row["Element"],
                "Predicted Structure": pred_structure,
                "OQMD Stable Structure": row["DFT Stable Structure"],
                "Actual Structure": row["Actual Structure"],
                "Pred_E_BCC_rel_FCC": energies["BCC"],
                "Pred_E_FCC_rel_FCC": energies["FCC"],
                "Pred_E_HCP_rel_FCC": energies["HCP"],
                "Matches OQMD": pred_structure == row["DFT Stable Structure"],
                "Matches Actual": pred_structure == row["Actual Structure"],
            }
        )
    return pd.DataFrame(rows)


def structure_accuracy_tables(df: pd.DataFrame, x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_frames = []
    for model_name, model in make_models().items():
        pred_df = predict_structures_from_pair_models(df, x, model, LeaveOneOut())
        prediction_frames.append(pred_df.assign(Model=model_name))
        rows.append(
            {
                "Model": model_name,
                "Accuracy_vs_OQMD": accuracy_score(
                    pred_df["OQMD Stable Structure"], pred_df["Predicted Structure"]
                ),
                "Accuracy_vs_Actual": accuracy_score(
                    pred_df["Actual Structure"], pred_df["Predicted Structure"]
                ),
                "Wrong_vs_OQMD": ", ".join(
                    pred_df.loc[~pred_df["Matches OQMD"], "Element"].tolist()
                ),
                "Wrong_vs_Actual": ", ".join(
                    pred_df.loc[~pred_df["Matches Actual"], "Element"].tolist()
                ),
            }
        )
    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


def fit_final_equations(df: pd.DataFrame, x: pd.DataFrame, model_name: str = "Ridge Regression") -> pd.DataFrame:
    model = make_models()[model_name]
    rows = []
    for target_name, (a, b) in PAIR_TARGETS.items():
        y = (df[a] - df[b]).to_numpy()
        fitted = clone(model).fit(x, y)
        if "model" in fitted.named_steps and hasattr(fitted.named_steps["model"], "coef_"):
            coef = fitted.named_steps["model"].coef_
            intercept = fitted.named_steps["model"].intercept_
            rows.append({"Target": target_name, "Term": "Intercept", "Coefficient": intercept})
            for feature, value in zip(FEATURES, coef):
                rows.append({"Target": target_name, "Term": f"standardized {feature}", "Coefficient": value})
            if hasattr(fitted.named_steps["model"], "alpha_"):
                rows.append(
                    {
                        "Target": target_name,
                        "Term": "selected alpha",
                        "Coefficient": fitted.named_steps["model"].alpha_,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES].copy()

    vif = vif_table(x)
    target_cv = target_cv_table(df, x)
    structure_acc, structure_predictions = structure_accuracy_tables(df, x)
    final_equations = fit_final_equations(df, x, "Ridge Regression")

    vif.to_csv(OUT_DIR / "vif_collinearity.csv", index=False, encoding="utf-8-sig")
    target_cv.to_csv(OUT_DIR / "loocv_target_regression_metrics.csv", index=False, encoding="utf-8-sig")
    structure_acc.to_csv(OUT_DIR / "loocv_structure_accuracy.csv", index=False, encoding="utf-8-sig")
    structure_predictions.to_csv(OUT_DIR / "loocv_structure_predictions.csv", index=False, encoding="utf-8-sig")
    final_equations.to_csv(OUT_DIR / "ridge_final_equations.csv", index=False, encoding="utf-8-sig")

    ridge_pred = structure_predictions[structure_predictions["Model"] == "Ridge Regression"]
    oqmd_labels = ["BCC", "FCC", "HCP"]
    cm_oqmd = pd.DataFrame(
        confusion_matrix(
            ridge_pred["OQMD Stable Structure"],
            ridge_pred["Predicted Structure"],
            labels=oqmd_labels,
        ),
        index=[f"OQMD {x}" for x in oqmd_labels],
        columns=[f"Pred {x}" for x in oqmd_labels],
    )
    cm_actual = pd.DataFrame(
        confusion_matrix(
            ridge_pred["Actual Structure"],
            ridge_pred["Predicted Structure"],
            labels=oqmd_labels,
        ),
        index=[f"Actual {x}" for x in oqmd_labels],
        columns=[f"Pred {x}" for x in oqmd_labels],
    )
    cm_oqmd.to_csv(OUT_DIR / "ridge_confusion_matrix_vs_oqmd.csv", encoding="utf-8-sig")
    cm_actual.to_csv(OUT_DIR / "ridge_confusion_matrix_vs_actual.csv", encoding="utf-8-sig")

    print(f"Saved regression outputs to {OUT_DIR}")
    print("\nLOOCV target regression metrics:")
    print(target_cv.sort_values(["Target", "RMSE"]).to_string(index=False))
    print("\nLOOCV structure accuracy:")
    print(structure_acc.to_string(index=False))
    print("\nRidge confusion matrix vs OQMD:")
    print(cm_oqmd.to_string())
    print("\nRidge confusion matrix vs actual:")
    print(cm_actual.to_string())


if __name__ == "__main__":
    main()
