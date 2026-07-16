from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "output" / "delta_e_transition_dataset.csv"
OUT_DIR = ROOT / "output" / "report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": ["Malgun Gothic", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["DFT Matches Actual"] = df["DFT Stable Structure"] == df["Actual Structure"]
    return df


def save_delta_distribution(df: pd.DataFrame) -> None:
    delta = df["Delta_E"].dropna()
    median = delta.median()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [1.25, 1]},
        constrained_layout=True,
    )

    sns.histplot(delta, bins=12, ax=axes[0], color="#7EA6C8", edgecolor="white")
    axes[0].axvline(median, color="#1F2933", linestyle="--", linewidth=1.8)
    axes[0].axvspan(0, 0.01, color="#FDE68A", alpha=0.35, label="very small ΔE (<0.01)")
    axes[0].axvspan(0.01, 0.10, color="#BFDBFE", alpha=0.25, label="moderate ΔE (0.01-0.10)")
    axes[0].set_title("Delta_E Distribution: Full Range")
    axes[0].set_xlabel("Delta_E (eV/atom)")
    axes[0].set_ylabel("Number of elements")
    axes[0].text(
        median,
        axes[0].get_ylim()[1] * 0.92,
        f"median = {median:.4f}",
        ha="left",
        va="top",
        fontsize=10,
        color="#1F2933",
    )
    axes[0].legend(frameon=True, loc="upper right")

    zoom = df[df["Delta_E"] <= 0.12].copy()
    zoom = zoom.sort_values("Delta_E")
    colors = np.where(zoom["DFT Matches Actual"], "#4E79A7", "#E15759")
    axes[1].barh(zoom["Element"], zoom["Delta_E"], color=colors, edgecolor="white")
    axes[1].axvline(0.01, color="#F59E0B", linestyle="--", linewidth=1.5)
    axes[1].axvline(0.10, color="#6B7280", linestyle=":", linewidth=1.5)
    axes[1].set_title("Zoom: Low Delta_E Region")
    axes[1].set_xlabel("Delta_E (eV/atom)")
    axes[1].set_ylabel("")
    axes[1].invert_yaxis()

    for _, row in zoom.loc[~zoom["DFT Matches Actual"]].iterrows():
        axes[1].text(
            row["Delta_E"] + 0.002,
            list(zoom["Element"]).index(row["Element"]),
            "mismatch",
            va="center",
            fontsize=8,
            color="#B91C1C",
        )

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", label="DFT = actual", markerfacecolor="#4E79A7", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="DFT != actual", markerfacecolor="#E15759", markersize=10),
    ]
    axes[1].legend(handles=legend_handles, frameon=True, loc="lower right")

    fig.suptitle("Small Delta_E values indicate strong structure competition", y=1.03)
    fig.savefig(OUT_DIR / "01_delta_e_distribution_with_zoom.png", bbox_inches="tight")
    plt.close(fig)


def save_ranked_delta(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("Delta_E").copy()
    plot_df["Transition Label"] = np.where(plot_df["Has Transition"].astype(str) == "True", "phase transition", "no reported transition")

    colors = np.where(plot_df["DFT Matches Actual"], "#6BAED6", "#E15759")
    edgecolors = np.where(plot_df["Transition Label"] == "phase transition", "#111827", "white")
    linewidths = np.where(plot_df["Transition Label"] == "phase transition", 1.4, 0.4)

    fig, ax = plt.subplots(figsize=(9, 8.5), constrained_layout=True)
    y = np.arange(len(plot_df))
    ax.barh(
        y,
        plot_df["Delta_E"],
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Element"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 0.7)
    ax.axvline(0.01, color="#F59E0B", linestyle="--", linewidth=1.6)
    ax.axvline(0.10, color="#6B7280", linestyle=":", linewidth=1.8)
    ax.text(0.0105, -1.0, "0.01", color="#B45309", fontsize=9)
    ax.text(0.105, -1.0, "0.10", color="#374151", fontsize=9)
    ax.set_xlabel("Delta_E (eV/atom, log scale)")
    ax.set_ylabel("Element")
    ax.set_title("Elements Ranked by Delta_E")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        if not row["DFT Matches Actual"]:
            ax.text(row["Delta_E"] * 1.2, i, "DFT mismatch", va="center", fontsize=8, color="#B91C1C")
        if str(row["Has Transition"]) == "True":
            ax.scatter(row["Delta_E"], i, s=55, facecolors="none", edgecolors="#111827", linewidths=1.4, zorder=3)

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#6BAED6", label="DFT = actual", markersize=10),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#E15759", label="DFT != actual", markersize=10),
        Line2D([0], [0], marker="o", color="#111827", markerfacecolor="none", label="phase transition", markersize=8, linestyle="None"),
        Line2D([0], [0], color="#F59E0B", linestyle="--", label="very small ΔE threshold"),
    ]
    ax.legend(handles=handles, frameon=True, loc="lower right")

    fig.savefig(OUT_DIR / "02_elements_ranked_by_delta_e_log.png", bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(df: pd.DataFrame) -> None:
    labels = ["BCC", "FCC", "HCP"]
    valid = df.dropna(subset=["Actual Structure", "DFT Stable Structure"])
    acc = (valid["Actual Structure"] == valid["DFT Stable Structure"]).mean()
    cm = confusion_matrix(valid["Actual Structure"], valid["DFT Stable Structure"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"DFT {x}" for x in labels])

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor="white", ax=ax)
    ax.set_title(f"Actual Structure vs DFT Stable Structure\nAccuracy = {acc:.1%} ({int(acc * len(valid))}/{len(valid)})")
    ax.set_xlabel("Predicted stable structure from DFT relative energy")
    ax.set_ylabel("Actual room-temperature structure")
    fig.savefig(OUT_DIR / "03_confusion_matrix_readable.png", bbox_inches="tight")
    plt.close(fig)


def save_fe_case(df: pd.DataFrame) -> None:
    fe = df[df["Element"] == "Fe"].iloc[0]
    energies = pd.DataFrame(
        {
            "Structure": ["BCC", "HCP", "FCC"],
            "Relative Energy": [fe["E_BCC"], fe["E_HCP"], fe["E_FCC"]],
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    colors = ["#59A14F", "#9CA3AF", "#F28E2B"]
    ax.bar(energies["Structure"], energies["Relative Energy"], color=colors, edgecolor="white")
    ax.set_ylabel("Relative Energy (eV/atom)")
    ax.set_xlabel("Structure")
    ax.set_title("Fe Case: DFT favors BCC at 0 K, but Fe transforms to FCC at high T")
    ax.annotate(
        "room-temperature\nactual structure",
        xy=(0, fe["E_BCC"]),
        xytext=(0.2, 0.045),
        arrowprops={"arrowstyle": "->", "color": "#166534"},
        fontsize=10,
        color="#166534",
    )
    ax.annotate(
        "912 °C transition:\nBCC → FCC",
        xy=(2, fe["E_FCC"]),
        xytext=(1.35, 0.13),
        arrowprops={"arrowstyle": "->", "color": "#9A3412"},
        fontsize=10,
        color="#9A3412",
    )
    fig.savefig(OUT_DIR / "04_fe_case_annotated.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    df = load_data()
    save_delta_distribution(df)
    save_ranked_delta(df)
    save_confusion_matrix(df)
    save_fe_case(df)
    print(f"Saved report-ready figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
