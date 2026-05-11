"""
plot_results.py
===============
Reproduit le style de figure du tuto (4 courbes LB/UB conservative + incrémentale,
gaps colorés, région certifiée hachurée) en ajoutant des bandes de percentiles
(P10–P90) pour les répétitions.

Une figure par (lambda, variante) → 12 figures sauvegardées dans figures/.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Config
# =============================================================================

RESULTS_DIR = Path("results_sweep")
OUTPUT_DIR  = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDAS        = [0.0] #, 0.25, 0.5, 1.0]
SENSOR_BUDGETS = [5, 10, 15, 20, 25]
P_LOW, P_HIGH  = 10, 90

# Couleurs identiques au tuto
COLOR_CONS_LB = "#1f77b4"   # bleu
COLOR_CONS_UB = "#ff7f0e"   # orange
COLOR_INC_LB  = "#2ca02c"   # vert
COLOR_INC_UB  = "#d62728"   # rouge

VAR_LABELS = {
    "fd"      : r"FD Jacobians  ($\Sigma_{\rm signal}$)",
    "free"    : r"Linear regression  ($\Sigma_{\rm signal}^{\rm free}$)",
    "free_nn" : r"Neural network  ($\Sigma_{\rm signal}^{\rm free,\,NN}$)",
}

# =============================================================================
# Chargement
# =============================================================================

def load_lambda(lam: float) -> dict | None:
    path = RESULTS_DIR / f"results_lambda_{lam:.2f}.npz"
    if not path.exists():
        print(f"  [warning] fichier manquant : {path}")
        return None
    d = np.load(path)
    out = {
        "lambda_"       : float(d["lambda_val"]),
        "eig_offset"    : float(d["eig_offset"]),
        "sensor_budgets": d["sensor_budgets"].tolist(),
    }
    for sel in ("inc", "cons"):
        for var in ("fd", "free", "free_nn"):
            for bound in ("lb", "ub"):
                key = f"{sel}_{var}_{bound}"
                out[key] = d[key]   # (n_repeats, n_budgets)
    return out


def load_all() -> dict[float, dict]:
    return {lam: d for lam in LAMBDAS if (d := load_lambda(lam)) is not None}


# =============================================================================
# Tracé d'une figure (lambda, variante)
# =============================================================================

def plot_one(ax, data: dict, var: str):
    """
    Reproduit le style du tuto sur ax :
      - LB conservative  : bleu  (trait plein, marqueur rond)
      - UB conservative  : orange (tirets, marqueur rond)
      - LB incrémentale  : vert  (trait plein, marqueur rond)
      - UB incrémentale  : rouge (trait plein, marqueur carré)
      - Gap conservative : bleu clair semi-transparent
      - Gap incrémental  : saumon semi-transparent
      - Région certifiée : vert hachuré
    Bandes P10-P90 autour de chaque médiane.
    """
    budgets = np.asarray(data["sensor_budgets"])
    eig_off = data["eig_offset"]

    def med_band(key):
        arr = data[key]   # (n_repeats, n_budgets)
        return (
            np.median(arr, axis=0),
            np.percentile(arr, P_LOW,  axis=0),
            np.percentile(arr, P_HIGH, axis=0),
        )

    med_cons_lb, lo_cons_lb, hi_cons_lb = med_band(f"cons_{var}_lb")
    med_cons_ub, lo_cons_ub, hi_cons_ub = med_band(f"cons_{var}_ub")
    med_inc_lb,  lo_inc_lb,  hi_inc_lb  = med_band(f"inc_{var}_lb")
    med_inc_ub,  lo_inc_ub,  hi_inc_ub  = med_band(f"inc_{var}_ub")

    # ---- gaps (sur les médianes) ----
    ax.fill_between(budgets, med_cons_lb, med_cons_ub,
                    color=COLOR_CONS_LB, alpha=0.15, linewidth=0)
    ax.fill_between(budgets, med_inc_lb, med_inc_ub,
                    color=COLOR_INC_UB, alpha=0.15, linewidth=0)

    # ---- région certifiée = intersection des deux intervalles (médianes) ----
    cert_lo = np.maximum(med_cons_lb, med_inc_lb)
    cert_hi = np.minimum(med_cons_ub, med_inc_ub)
    mask = cert_lo <= cert_hi
    if mask.any():
        ax.fill_between(budgets, cert_lo, cert_hi,
                        where=mask,
                        color=COLOR_INC_LB, alpha=0.30,
                        hatch="////", linewidth=0,
                        label="Common certified region")

    # ---- bandes percentiles (P10-P90) autour de chaque courbe ----
    ax.fill_between(budgets, lo_cons_lb, hi_cons_lb,
                    color=COLOR_CONS_LB, alpha=0.12, linewidth=0)
    ax.fill_between(budgets, lo_cons_ub, hi_cons_ub,
                    color=COLOR_CONS_UB, alpha=0.12, linewidth=0)
    ax.fill_between(budgets, lo_inc_lb,  hi_inc_lb,
                    color=COLOR_INC_LB,  alpha=0.12, linewidth=0)
    ax.fill_between(budgets, lo_inc_ub,  hi_inc_ub,
                    color=COLOR_INC_UB,  alpha=0.12, linewidth=0)

    # ---- courbes médianes ----
    ax.plot(budgets, med_cons_lb, color=COLOR_CONS_LB, linewidth=2.0,
            marker="o", markersize=5,
            label="Lower bound (conservative)")
    ax.plot(budgets, med_cons_ub, color=COLOR_CONS_UB, linewidth=2.0,
            marker="o", markersize=5, linestyle="--",
            label="Upper bound (conservative)")
    ax.plot(budgets, med_inc_lb,  color=COLOR_INC_LB,  linewidth=2.0,
            marker="o", markersize=5,
            label="Lower bound (incremental)")
    ax.plot(budgets, med_inc_ub,  color=COLOR_INC_UB,  linewidth=2.0,
            marker="s", markersize=5,
            label="Upper bound (incremental)")

    # ---- ligne de référence eig_offset ----
    ax.axhline(eig_off, color="gray", linewidth=1.0,
               linestyle=":", alpha=0.8,
               label=f"EIG reference ({eig_off:.2f})")

    ax.set_xlabel("Number of sensors", fontsize=10)
    ax.set_ylabel("Information gain", fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc="lower right")


# =============================================================================
# Génération des figures
# =============================================================================

def make_figures(all_data: dict[float, dict]):
    vars_ = ["fd", "free", "free_nn"]

    for lam, data in all_data.items():
        n_rep = data["inc_fd_lb"].shape[0]
        for var in vars_:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.suptitle(
                "EIG bounds, gaps, and common certified region\n"
                rf"$\lambda = {lam}$ — {VAR_LABELS[var]} "
                f"[median + P{P_LOW}–P{P_HIGH}, {n_rep} repeats]",
                fontsize=10,
            )
            plot_one(ax, data, var)
            fig.tight_layout()

            fname = f"bounds_lambda_{lam:.2f}_{var}.pdf"
            path  = OUTPUT_DIR / fname
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            print(f"  {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Chargement des données...")
    all_data = load_all()

    if not all_data:
        print("Aucun fichier trouvé dans results/. Lance d'abord run_experiment.py.")
        return

    print(f"Lambdas disponibles : {sorted(all_data.keys())}")
    n_rep = next(iter(all_data.values()))["inc_fd_lb"].shape[0]
    print(f"Nombre de répétitions : {n_rep}\n")

    make_figures(all_data)
    print(f"\nFigures sauvegardées dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
