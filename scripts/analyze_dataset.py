from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


FEATURES = [
    "Atomic Radius pm",
    "Electronegativity",
    "First Ionization Energy kJ mol-1",
    "d Electrons",
    "Density g cm-3",
    "Melting Point C",
]


def save_confusion_matrix(df: pd.DataFrame, fig_dir: Path) -> None:
    valid = df.dropna(subset=["Actual Structure", "DFT Stable Structure"])
    labels = ["BCC", "FCC", "HCP"]
    matrix = confusion_matrix(valid["Actual Structure"], valid["DFT Stable Structure"], labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("DFT Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def save_delta_plots(df: pd.DataFrame, fig_dir: Path) -> None:
    valid = df.dropna(subset=["Delta_E"])
    plt.figure(figsize=(6, 4))
    sns.histplot(valid["Delta_E"], bins=12, kde=True)
    plt.xlabel("Delta E (eV/atom)")
    plt.tight_layout()
    plt.savefig(fig_dir / "delta_e_histogram.png", dpi=200)
    plt.close()

    for feature in ["d Electrons", "Atomic Radius pm"]:
        data = df.dropna(subset=["Delta_E", feature])
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=data, x=feature, y="Delta_E", hue="Has Transition")
        plt.tight_layout()
        plt.savefig(fig_dir / f"delta_e_vs_{feature.lower().replace(' ', '_')}.png", dpi=200)
        plt.close()


def save_energy_comparison(df: pd.DataFrame, fig_dir: Path) -> None:
    long = df.melt(
        id_vars=["Element"],
        value_vars=["E_BCC", "E_FCC", "E_HCP"],
        var_name="Structure",
        value_name="Energy eV atom-1",
    ).dropna()
    long["Structure"] = long["Structure"].str.replace("E_", "", regex=False)
    plt.figure(figsize=(max(8, len(df) * 0.35), 5))
    sns.barplot(data=long, x="Element", y="Energy eV atom-1", hue="Structure")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_dir / "structure_energy_comparison.png", dpi=200)
    plt.close()


def save_feature_importance(df: pd.DataFrame, fig_dir: Path) -> None:
    model_df = df.dropna(subset=FEATURES + ["Delta_E"])
    if len(model_df) < 8:
        print("[WARN] Not enough complete rows for Random Forest feature importance.")
        return
    x = model_df[FEATURES]
    y = model_df["Delta_E"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
    importance = importance.sort_values("Importance", ascending=False)
    importance.to_csv(fig_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 4))
    sns.barplot(data=importance, x="Importance", y="Feature")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", dpi=200)
    plt.close()
    print(f"Random Forest R^2 train={model.score(x_train, y_train):.3f}, test={model.score(x_test, y_test):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze final DFT structure dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--fig-dir", default="output/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    valid = df.dropna(subset=["Actual Structure", "DFT Stable Structure"])
    accuracy = (valid["Actual Structure"] == valid["DFT Stable Structure"]).mean() if len(valid) else float("nan")
    transition_summary = df.groupby("Has Transition", dropna=False)["Delta_E"].describe()
    transition_summary.to_csv(fig_dir / "transition_delta_e_summary.csv", encoding="utf-8-sig")

    save_confusion_matrix(df, fig_dir)
    save_delta_plots(df, fig_dir)
    save_energy_comparison(df, fig_dir)
    save_feature_importance(df, fig_dir)

    print(f"Accuracy: {accuracy:.3f}" if pd.notna(accuracy) else "Accuracy: NA")
    print(f"Saved figures and summary files to {fig_dir}")


if __name__ == "__main__":
    main()
