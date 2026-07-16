from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OQMD_URL = "https://oqmd.org/oqmdapi/formationenergy"
STRUCTURES = ("BCC", "FCC", "HCP")


STRUCTURE_HINTS = {
    "BCC": (
        "bcc",
        "body centered cubic",
        "body-centered cubic",
        "im-3m",
        "im3m",
        "alpha-fe",
        "spacegroup 229",
        "space group 229",
    ),
    "FCC": (
        "fcc",
        "face centered cubic",
        "face-centered cubic",
        "fm-3m",
        "fm3m",
        "spacegroup 225",
        "space group 225",
    ),
    "HCP": (
        "hcp",
        "hexagonal close packed",
        "hexagonal close-packed",
        "p63/mmc",
        "p6_3/mmc",
        "spacegroup 194",
        "space group 194",
    ),
}


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def clean_symbol(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty element symbol")
    return value[0].upper() + value[1:].lower()


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_json(nested, full_key))
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            full_key = f"{prefix}.{idx}" if prefix else str(idx)
            items.update(flatten_json(nested, full_key))
    else:
        items[prefix] = value
    return items


def first_present(flat: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    lowered = {key.lower(): value for key, value in flat.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for key, value in lowered.items():
        if any(key.endswith(f".{candidate.lower()}") for candidate in candidates):
            return value
    return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def infer_structure(record: dict[str, Any]) -> str | None:
    flat = flatten_json(record)
    preferred_fragments = []
    preferred_keys = (
        "name",
        "prototype",
        "label",
        "structure",
        "crystal_structure",
        "spacegroup",
        "space_group",
        "space group",
        "sg",
    )
    for key, value in flat.items():
        key_lower = key.lower()
        if any(preferred_key in key_lower for preferred_key in preferred_keys):
            preferred_fragments.append(f"{key_lower} {value}")
    fragments = preferred_fragments or [str(value) for value in flat.values() if value is not None]
    text = " ".join(str(value).lower() for value in fragments if value is not None)
    text = re.sub(r"[_\-]+", " ", text)

    scores: dict[str, int] = {}
    for structure, hints in STRUCTURE_HINTS.items():
        scores[structure] = sum(1 for hint in hints if hint.lower().replace("-", " ") in text)

    best_structure = max(scores, key=scores.get)
    return best_structure if scores[best_structure] > 0 else None


def extract_energy_per_atom(
    record: dict[str, Any],
    allow_formation_energy_fallback: bool = False,
) -> tuple[float | None, str | None]:
    flat = flatten_json(record)
    direct_candidates = (
        "energy_per_atom",
        "energy_pa",
        "energyperatom",
        "total_energy_per_atom",
        "totalenergy_per_atom",
        "e_per_atom",
    )
    direct_energy = as_float(
        first_present(
            flat,
            direct_candidates,
        )
    )
    if direct_energy is not None:
        return direct_energy, "energy_per_atom"

    total_energy = as_float(
        first_present(flat, ("energy", "total_energy", "totalenergy", "final_energy"))
    )
    natoms = as_float(first_present(flat, ("natoms", "num_atoms", "n_atoms", "nsites")))
    if total_energy is not None and natoms and natoms > 0:
        return total_energy / natoms, "total_energy_divided_by_natoms"

    if allow_formation_energy_fallback:
        fallback = as_float(first_present(flat, ("delta_e", "formationenergy", "formation_energy")))
        if fallback is not None:
            return fallback, "formation_energy_or_delta_e_fallback"
    return None, None


def extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "results", "response", "entries"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def fetch_oqmd_records(
    element: str,
    endpoint: str,
    limit: int,
    timeout: int,
    delay: float,
    retries: int,
    cache_dir: Path | None,
    refresh_cache: bool,
    quiet: bool,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"{element}.json" if cache_dir else None
    if cache_path and cache_path.exists() and not refresh_cache:
        log(f"[CACHE] {element}: using cached OQMD response ({cache_path})", quiet)
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        records = extract_records(payload)
        log(f"[OK] {element}: loaded {len(records)} raw records from cache", quiet)
        return records

    params = {"composition": element, "limit": limit}
    last_error: requests.RequestException | None = None
    for attempt in range(1, retries + 1):
        try:
            log(f"[LOAD] {element}: requesting OQMD ({attempt}/{retries})", quiet)
            response = requests.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            records = extract_records(payload)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                log(f"[CACHE] {element}: saved response to {cache_path}", quiet)
            if delay:
                time.sleep(delay)
            log(f"[OK] {element}: loaded {len(records)} raw records from OQMD", quiet)
            return records
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = delay * attempt if delay else attempt
                log(f"[WARN] {element}: request failed ({attempt}/{retries}) - {exc}", quiet)
                log(f"[WAIT] {element}: retrying after {wait_seconds:.1f}s", quiet)
                time.sleep(wait_seconds)
            else:
                break
    if last_error:
        raise last_error
    return []


def records_to_energy_rows(
    element: str,
    records: list[dict[str, Any]],
    allow_formation_energy_fallback: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        structure = infer_structure(record)
        energy, source = extract_energy_per_atom(record, allow_formation_energy_fallback)
        if structure not in STRUCTURES or energy is None:
            continue
        flat = flatten_json(record)
        rows.append(
            {
                "Element": element,
                "Structure": structure,
                "Energy eV atom-1": energy,
                "Energy Source Field": source,
                "OQMD Entry": first_present(flat, ("entry_id", "id", "calculation_id")),
                "OQMD Raw Name": first_present(flat, ("name", "prototype", "label")),
            }
        )
    return rows


def choose_lowest_energy(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["Element", "E_BCC", "E_FCC", "E_HCP"])
    best = (
        rows.sort_values(["Element", "Structure", "Energy eV atom-1"])
        .groupby(["Element", "Structure"], as_index=False)
        .first()
    )
    wide = best.pivot(index="Element", columns="Structure", values="Energy eV atom-1").reset_index()
    for structure in STRUCTURES:
        if structure not in wide.columns:
            wide[structure] = np.nan
    wide = wide.rename(columns={structure: f"E_{structure}" for structure in STRUCTURES})
    return wide[["Element", "E_BCC", "E_FCC", "E_HCP"]]


def add_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    energy_cols = ["E_BCC", "E_FCC", "E_HCP"]
    structures = ["BCC", "FCC", "HCP"]

    stable_structures = []
    deltas = []
    second_structures = []
    for _, row in df.iterrows():
        values = [(structure, as_float(row[col])) for structure, col in zip(structures, energy_cols)]
        values = [(structure, energy) for structure, energy in values if energy is not None]
        values.sort(key=lambda item: item[1])
        if not values:
            stable_structures.append(np.nan)
            second_structures.append(np.nan)
            deltas.append(np.nan)
        elif len(values) == 1:
            stable_structures.append(values[0][0])
            second_structures.append(np.nan)
            deltas.append(np.nan)
        else:
            stable_structures.append(values[0][0])
            second_structures.append(values[1][0])
            deltas.append(values[1][1] - values[0][1])

    df["DFT Stable Structure"] = stable_structures
    df["Second Stable Structure"] = second_structures
    df["Delta_E"] = deltas
    df["Delta_E_BCC_FCC"] = df["E_BCC"] - df["E_FCC"]
    df["Delta_E_BCC_HCP"] = df["E_BCC"] - df["E_HCP"]
    df["Delta_E_FCC_HCP"] = df["E_FCC"] - df["E_HCP"]
    return df


def load_reference_data(reference_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actual = pd.read_csv(reference_dir / "actual_structures.csv")
    props = pd.read_csv(reference_dir / "element_properties.csv")
    transitions = pd.read_csv(reference_dir / "phase_transitions.csv")
    for frame in (actual, props, transitions):
        frame["Element"] = frame["Element"].map(clean_symbol)
    return actual, props, transitions


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    elements = [clean_symbol(element) for element in args.elements]
    reference_dir = Path(args.reference_dir)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else raw_dir / "oqmd_cache"

    all_rows: list[dict[str, Any]] = []
    failed_elements = []
    if args.mock_json:
        payload = json.loads(Path(args.mock_json).read_text(encoding="utf-8"))
        for element in elements:
            log(f"[LOAD] {element}: reading mock JSON", args.quiet)
            records = extract_records(payload.get(element, payload) if isinstance(payload, dict) else payload)
            rows = records_to_energy_rows(
                element,
                records,
                allow_formation_energy_fallback=args.allow_formation_energy_fallback,
            )
            all_rows.extend(rows)
            counts = pd.Series([row["Structure"] for row in rows]).value_counts().to_dict() if rows else {}
            log(f"[OK] {element}: extracted {len(rows)} usable BCC/FCC/HCP energy rows {counts}", args.quiet)
    else:
        for idx, element in enumerate(elements, start=1):
            log(f"[{idx}/{len(elements)}] {element}: start", args.quiet)
            try:
                records = fetch_oqmd_records(
                    element,
                    endpoint=args.oqmd_url,
                    limit=args.limit,
                    timeout=args.timeout,
                    delay=args.delay,
                    retries=args.retries,
                    cache_dir=cache_dir,
                    refresh_cache=args.refresh_cache,
                    quiet=args.quiet,
                )
            except requests.RequestException as exc:
                log(f"[FAIL] {element}: OQMD request failed after {args.retries} attempt(s) - {exc}", args.quiet)
                failed_elements.append(element)
                records = []
            rows = records_to_energy_rows(
                element,
                records,
                allow_formation_energy_fallback=args.allow_formation_energy_fallback,
            )
            all_rows.extend(rows)
            counts = pd.Series([row["Structure"] for row in rows]).value_counts().to_dict() if rows else {}
            if rows:
                log(f"[OK] {element}: extracted {len(rows)} usable BCC/FCC/HCP energy rows {counts}", args.quiet)
            else:
                log(f"[WARN] {element}: no usable BCC/FCC/HCP energy rows extracted", args.quiet)

    raw_rows = pd.DataFrame(all_rows)
    raw_rows.to_csv(raw_dir / "oqmd_structure_energy_rows.csv", index=False, encoding="utf-8-sig")
    if failed_elements:
        (raw_dir / "failed_elements.txt").write_text("\n".join(failed_elements), encoding="utf-8")
        print(f"[WARN] Failed elements saved to {raw_dir / 'failed_elements.txt'}")

    energy = choose_lowest_energy(raw_rows)
    actual, props, transitions = load_reference_data(reference_dir)
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
    parser = argparse.ArgumentParser(description="Build BCC/FCC/HCP DFT energy dataset from OQMD.")
    parser.add_argument("--elements", nargs="+", required=True, help="Element symbols to query.")
    parser.add_argument("--oqmd-url", default=DEFAULT_OQMD_URL, help="OQMD API endpoint.")
    parser.add_argument("--limit", type=int, default=200, help="Max OQMD records per element.")
    parser.add_argument("--timeout", type=int, default=90, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per element.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds.")
    parser.add_argument("--cache-dir", help="Directory for per-element OQMD JSON cache.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore existing cached OQMD JSON.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-element progress logs.")
    parser.add_argument("--reference-dir", default=PROJECT_ROOT / "data" / "reference")
    parser.add_argument("--raw-dir", default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output", default=PROJECT_ROOT / "output" / "final_dataset.csv")
    parser.add_argument("--mock-json", help="Local OQMD-like JSON for parser testing.")
    parser.add_argument(
        "--allow-formation-energy-fallback",
        action="store_true",
        help="Use formation energy or delta_e only when true electronic energy is not present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset(args)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    complete = df[["E_BCC", "E_FCC", "E_HCP"]].notna().all(axis=1).sum()
    print(f"Saved {len(df)} rows to {output}")
    print(f"Rows with all BCC/FCC/HCP energies: {complete}")


if __name__ == "__main__":
    main()
