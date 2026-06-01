"""
save_csv.py
===========
Utilitaire pour gérer les résultats EIG.

Modes disponibles :
- conversion d'un fichier results_lambda_X.XX.npz en CSVs
- construction d'un fichier results_lambda_X.XX.npz à partir de fichiers .npy

Structure de sortie CSV : un fichier par clé de tableau du .npz
  lambda_0.00_cons_fd_lb.csv
  lambda_0.00_cons_fd_ub.csv
  ...

Structure .npz attendue :
  eig_offset, sensor_budgets, lambda_val, n_repeats,
  cons_fd_lb, cons_fd_ub, cons_fd_inc_lb, cons_fd_inc_ub,
  cons_free_lb, cons_free_ub, cons_free_inc_lb, cons_free_inc_ub,
  cons_nn_lb, cons_nn_ub, cons_nn_inc_lb, cons_nn_inc_ub,
  inc_fd_lb, inc_fd_ub, inc_fd_inc_lb, inc_fd_inc_ub,
  inc_free_lb, inc_free_ub, inc_free_inc_lb, inc_free_inc_ub,
  inc_nn_lb, inc_nn_ub, inc_nn_inc_lb, inc_nn_inc_ub
"""

from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results_sweep")
CSV_DIR = Path("csv")
CSV_DIR.mkdir(parents=True, exist_ok=True)

# mapping (variante, sélection) -> clés npz
#   LBC = Lower Bound Conservative (BI)
#   UBC = Upper Bound Conservative (BS)
#   LBI = Lower Bound Incremental
#   UBI = Upper Bound Incremental
KEYS = {
    ("fd", "inc"): dict(
        LBC="cons_fd_inc_lb",
        UBC="cons_fd_inc_ub",
        LBI="inc_fd_lb",
        UBI="inc_fd_ub",
    ),
    ("fd", "cons"): dict(
        LBC="cons_fd_lb",
        UBC="cons_fd_ub",
        LBI="cons_fd_inc_lb",
        UBI="cons_fd_inc_ub",
    ),
    ("free", "inc"): dict(
        LBC="cons_free_inc_lb",
        UBC="cons_free_inc_ub",
        LBI="inc_free_lb",
        UBI="inc_free_ub",
    ),
    ("free", "cons"): dict(
        LBC="cons_free_lb",
        UBC="cons_free_ub",
        LBI="cons_free_inc_lb",
        UBI="cons_free_inc_ub",
    ),
    ("nn", "inc"): dict(
        LBC="cons_nn_inc_lb",
        UBC="cons_nn_inc_ub",
        LBI="inc_nn_lb",
        UBI="inc_nn_ub",
    ),
    ("nn", "cons"): dict(
        LBC="cons_nn_lb",
        UBC="cons_nn_ub",
        LBI="cons_nn_inc_lb",
        UBI="cons_nn_inc_ub",
    ),
}
ARRAY_KEYS = [
    "cons_fd_lb", "cons_fd_ub",
    "cons_fd_inc_lb", "cons_fd_inc_ub",
    "cons_free_lb", "cons_free_ub",
    "cons_free_inc_lb", "cons_free_inc_ub",
    "cons_nn_lb", "cons_nn_ub",
    "cons_nn_inc_lb", "cons_nn_inc_ub",
    "inc_fd_lb", "inc_fd_ub",
    "inc_fd_inc_lb", "inc_fd_inc_ub",
    "inc_free_lb", "inc_free_ub",
    "inc_free_inc_lb", "inc_free_inc_ub",
    "inc_nn_lb", "inc_nn_ub",
    "inc_nn_inc_lb", "inc_nn_inc_ub",
]


def _scalar(item: np.ndarray):
    item = np.asarray(item)
    if item.shape == ():
        return item.item()
    if item.size == 1:
        return item.reshape(()).item()
    raise ValueError("Expected scalar array for this field")


def npz_to_csvs(path: Path):
    with np.load(path) as d:
        if "sensor_budgets" not in d:
            raise KeyError(f"Missing sensor_budgets in {path}")
        budgets = list(np.atleast_1d(d["sensor_budgets"]).astype(int))
        lam_val = _scalar(d["lambda_val"]) if "lambda_val" in d else None

        print(f"Processing {path.name}: lambda={lam_val}, budgets={budgets}")

        ignored_keys = {"eig_offset", "sensor_budgets", "lambda_val", "n_repeats"}
        for key in sorted(d.keys()):
            if key in ignored_keys:
                continue
            arr = np.atleast_2d(d[key])
            if arr.shape[1] != len(budgets):
                raise ValueError(
                    f"Unexpected shape for {key} in {path}: got {arr.shape}, expected (_, {len(budgets)})"
                )

            cols = {str(bud): arr[:, bi] for bi, bud in enumerate(budgets)}
            df = pd.DataFrame(cols)
            out = CSV_DIR / f"lambda_{lam_val:.2f}_{key}.csv"
            df.to_csv(out, index=False)
            print(f"  wrote {out}  shape={df.shape}")


def npys_to_npz(input_dir: Path, output: Path, lam_val: float | None, budgets: list[int] | None, n_repeats: int | None):
    arrays = {}
    missing = []

    for key in ARRAY_KEYS:
        path = input_dir / f"{key}.npy"
        if not path.exists():
            missing.append(path)
        else:
            arrays[key] = np.load(path)

    if missing:
        raise FileNotFoundError(
            "Missing required .npy files for packing:\n" + "\n".join(str(p) for p in missing)
        )

    if lam_val is None:
        path = input_dir / "lambda_val.npy"
        if path.exists():
            lam_val = _scalar(np.load(path))
        else:
            raise ValueError("Missing lambda value: provide --lambda-val or lambda_val.npy")

    if budgets is None:
        path = input_dir / "sensor_budgets.npy"
        if path.exists():
            budgets = list(np.atleast_1d(np.load(path)).astype(int))
        else:
            raise ValueError("Missing sensor budgets: provide --sensor-budgets or sensor_budgets.npy")

    if n_repeats is None:
        path = input_dir / "n_repeats.npy"
        if path.exists():
            n_repeats = int(_scalar(np.load(path)))
        else:
            raise ValueError("Missing n_repeats: provide --n-repeats or n_repeats.npy")

    eig_offset = None
    path = input_dir / "eig_offset.npy"
    if path.exists():
        eig_offset = _scalar(np.load(path))

    output.parent.mkdir(parents=True, exist_ok=True)
    to_save = {
        "sensor_budgets": np.array(budgets, dtype=int),
        "lambda_val": np.float64(lam_val),
        "n_repeats": np.int64(n_repeats),
        **arrays,
    }
    if eig_offset is not None:
        to_save["eig_offset"] = np.float64(eig_offset)

    np.savez(output, **to_save)
    print(f"Saved .npz file: {output}")


def find_npz_files():
    return sorted(RESULTS_DIR.glob("results_lambda_*.npz"))


def main():
    parser = argparse.ArgumentParser(description="Utility to convert .npz to CSV or pack .npy files into .npz.")
    parser.add_argument("--pack", action="store_true", help="Pack .npy files into a results_lambda_X.XX.npz file.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing source .npy files when packing.")
    parser.add_argument("--output", type=Path, help="Destination .npz file when packing.")
    parser.add_argument("--lambda-val", type=float, help="Lambda value to use when packing.")
    parser.add_argument("--sensor-budgets", nargs="+", type=int, help="Sensor budgets when packing.")
    parser.add_argument("--n-repeats", type=int, help="Number of repeats when packing.")
    parser.add_argument(
        "lambdas",
        nargs="*",
        type=float,
        help="Optional lambda values to process for CSV conversion (e.g. 0.0 0.25). If omitted, convert all files.",
    )
    args = parser.parse_args()

    if args.pack:
        if args.output is None:
            if args.lambda_val is None:
                raise SystemExit("--output or --lambda-val is required when packing")
            args.output = RESULTS_DIR / f"results_lambda_{args.lambda_val:.2f}.npz"
        npys_to_npz(args.input_dir, args.output, args.lambda_val, args.sensor_budgets, args.n_repeats)
        return

    files = find_npz_files()
    if args.lambdas:
        wanted = {f"results_lambda_{lam:.2f}.npz" for lam in args.lambdas}
        files = [f for f in files if f.name in wanted]

    if not files:
        print("No .npz files found in", RESULTS_DIR)
        return

    for path in files:
        npz_to_csvs(path)

    print("Done.")


if __name__ == "__main__":
    main()
