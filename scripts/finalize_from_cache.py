from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_dataset import (
    PROJECT_ROOT,
    add_energy_features,
    choose_lowest_energy,
    clean_symbol,
    extract_records,
    load_reference_data,
    log,
    records_to_energy_rows,
)


DEFAULT_ELEMENTS = [
    "Fe",
    "Cu",
    "Ti",
    "Ni",
    "Co",
    "Zn",
    "Zr",
    "Mo",
    "W",
    "Cr",
    "V",
    "Nb",
    "Ta",
    "Hf",
    "Sc",
    "Y",
    "Ag",
    "Au",
    "Pt",
    "Pd",
    "Al",
    "Mg",
    "Ca",
    "Li",
    "Na",
    "K",
    "Ba",
    "Sr",
]


ANALYSIS_COLUMNS = [
    "Element",
    "Actual Structure",
    "DFT Stable Structure",
    "Second Stable Structure",
    "DFT Matches Actual",
    "E_BCC",
    "E_FCC",
    "E_HCP",
    "Delta_E",
    "Delta_E_BCC_FCC",
    "Delta_E_BCC_HCP",
    "Delta_E_FCC_HCP",
    "Has Transition",
    "Transition Temp C",
    "Before",
    "After",
    "Atomic Number",
    "Atomic Radius pm",
    "Atomic Mass",
    "Density g cm-3",
    "Melting Point C",
    "Electronegativity",
    "First Ionization Energy kJ mol-1",
    "Valence Electrons",
    "d Electrons",
]


def rows_from_cache(args: argparse.Namespace, elements: list[str]) -> pd.DataFrame:
    cache_dir = Path(args.cache_dir)
    rows = []
    missing = []
    for element in elements:
        cache_path = cache_dir / f"{element}.json"
        if not cache_path.exists():
            missing.append(element)
            log(f"[MISS] {element}: cache file not found ({cache_path})", args.quiet)
            continue
        log(f"[CACHE] {element}: reading {cache_path}", args.quiet)
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        records = extract_records(payload)
        element_rows = records_to_energy_rows(
            element,
            records,
            allow_formation_energy_fallback=not args.no_formation_energy_fallback,
        )
        counts = pd.Series([row["Structure"] for row in element_rows]).value_counts().to_dict() if element_rows else {}
        log(f"[OK] {element}: extracted {len(element_rows)} usable BCC/FCC/HCP energy rows {counts}", args.quiet)
        rows.extend(element_rows)

    raw_rows = pd.DataFrame(rows)
    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_rows.to_csv(raw_output, index=False, encoding="utf-8-sig")

    if missing:
        missing_output = raw_output.parent / "missing_cache_elements.txt"
        missing_output.write_text("\n".join(missing), encoding="utf-8")
        log(f"[WARN] Missing cache elements saved to {missing_output}", args.quiet)
    return raw_rows


def finalize_dataset(args: argparse.Namespace) -> pd.DataFrame:
    elements = [clean_symbol(element) for element in args.elements]
    raw_rows = rows_from_cache(args, elements)
    energy = choose_lowest_energy(raw_rows)
    actual, props, transitions = load_reference_data(Path(args.reference_dir))
    df = (
        pd.DataFrame({"Element": elements})
        .merge(energy, on="Element", how="left")
        .merge(actual, on="Element", how="left")
        .merge(props, on="Element", how="left")
        .merge(transitions, on="Element", how="left")
    )
    df = add_energy_features(df)
    df["DFT Matches Actual"] = df["DFT Stable Structure"] == df["Actual Structure"]
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final analysis CSV from cached OQMD JSON files.")
    parser.add_argument("--elements", nargs="+", default=DEFAULT_ELEMENTS)
    parser.add_argument("--cache-dir", default=PROJECT_ROOT / "data" / "raw" / "oqmd_cache")
    parser.add_argument("--reference-dir", default=PROJECT_ROOT / "data" / "reference")
    parser.add_argument("--output", default=PROJECT_ROOT / "output" / "final_dataset.csv")
    parser.add_argument("--analysis-output", default=PROJECT_ROOT / "output" / "delta_e_transition_dataset.csv")
    parser.add_argument("--raw-output", default=PROJECT_ROOT / "data" / "raw" / "oqmd_structure_energy_rows.csv")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-formation-energy-fallback",
        action="store_true",
        help="Do not use OQMD delta_e when true electronic energy is not present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = finalize_dataset(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    available_analysis_cols = [column for column in ANALYSIS_COLUMNS if column in df.columns]
    analysis_output = Path(args.analysis_output)
    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    df[available_analysis_cols].to_csv(analysis_output, index=False, encoding="utf-8-sig")

    complete = df[["E_BCC", "E_FCC", "E_HCP"]].notna().all(axis=1).sum()
    missing = df.loc[df[["E_BCC", "E_FCC", "E_HCP"]].isna().any(axis=1), "Element"].tolist()
    print(f"Saved final dataset: {output}")
    print(f"Saved Delta E / transition dataset: {analysis_output}")
    print(f"Rows with all BCC/FCC/HCP energies: {complete}/{len(df)}")
    if missing:
        print(f"Elements still missing at least one energy: {', '.join(missing)}")


if __name__ == "__main__":
    main()
