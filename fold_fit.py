"""
fold_fit.py — fit FOLD and first-order models to your BOD data
==============================================================

Usage:
    python fold_fit.py examples/example_bod_data.csv
    python fold_fit.py mydata.csv --flow 20000 --effluent 30 --out results

Input CSV (one row per measurement, any number of samples and time points):
    sample,time_day,bod_mg_L
    A,1,211
    A,2,322
    ...

Outputs (in the --out folder, default "results"):
    fold_results.csv     parameters, fit statistics and design for each sample
    <sample>.png         observed BOD with fitted first-order and FOLD curves
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from fold_bod import (first_order, fold, fit_first_order, fit_fold, at_bound,
                      goodness_of_fit, design_activated_sludge, design_aerated_lagoon)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit FOLD and first-order BOD models.")
    ap.add_argument("csv", help="input CSV with columns sample,time_day,bod_mg_L")
    ap.add_argument("--out", default="results", help="output folder (default: results)")
    ap.add_argument("--flow", type=float, default=35000.0, help="design flow, m3/d")
    ap.add_argument("--effluent", type=float, default=20.0, help="target effluent BOD, mg/L")
    ap.add_argument("--loading", type=float, default=500.0, help="ASP BOD loading, g/m3/d")
    ap.add_argument("--no-plots", action="store_true", help="skip the figures")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.csv)
    need = {"sample", "time_day", "bod_mg_L"}
    if not need.issubset(df.columns):
        sys.exit(f"Input file must have columns: {', '.join(sorted(need))}")
    df = df.dropna(subset=["bod_mg_L"])
    os.makedirs(a.out, exist_ok=True)
    design = {"Q": a.flow, "S_eff": a.effluent, "loading_rate": a.loading}

    rows = []
    for name, g in df.groupby("sample", sort=False):
        g = g.sort_values("time_day")
        t = g["time_day"].to_numpy(float)
        y = g["bod_mg_L"].to_numpy(float)
        L0f, kf = fit_first_order(t, y)
        p = fit_fold(t, y)
        gf = goodness_of_fit(y, first_order(t, L0f, kf), 2)
        gd = goodness_of_fit(y, fold(t, *p), 4)
        row = {"sample": name, "n_points": len(t),
               "L0_FO": L0f, "k_FO": kf,
               "L0_FOLD": p[0], "k_FOLD": p[1], "alpha": p[2], "gamma": p[3],
               **{f"{m}_FO": v for m, v in gf.items()},
               **{f"{m}_FOLD": v for m, v in gd.items()},
               "FOLD_at_limit": "; ".join(at_bound(t, y, p)) or "none"}
        for tag, fn in [("ASP", design_activated_sludge), ("Lagoon", design_aerated_lagoon)]:
            for model, (L0, k) in [("FO", (L0f, kf)), ("FOLD", (p[0], p[1]))]:
                for key, v in fn(L0, k, design).items():
                    row[f"{tag}_{key}_{model}"] = v
        rows.append(row)

        print(f"\n=== {name} ({len(t)} points) ===")
        print(f"  First-order : L0 = {L0f:8.2f}  k = {kf:.4f}"
              f"            RMSE = {gf['RMSE']:.2f}  R2 = {gf['R2']:.4f}  AIC = {gf['AIC']:.2f}")
        print(f"  FOLD        : L0 = {p[0]:8.2f}  k = {p[1]:.4f}  alpha = {p[2]:.4f}  gamma = {p[3]:.4f}"
              f"  RMSE = {gd['RMSE']:.2f}  R2 = {gd['R2']:.4f}  AIC = {gd['AIC']:.2f}")
        if row["FOLD_at_limit"] != "none":
            print(f"  Note: FOLD parameter(s) at fitting limit: {row['FOLD_at_limit']}"
                  " — treat these values with caution.")

        if not a.no_plots:
            _plot(name, t, y, (L0f, kf), p, a.out)

    out_csv = os.path.join(a.out, "fold_results.csv")
    pd.DataFrame(rows).round(4).to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")


def _plot(name, t, y, pf, p, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt = np.linspace(0, t.max() * 1.05, 200)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(t, y, "ko", label="Observed")
    ax.plot(tt, first_order(tt, *pf), "b--", label="First-order")
    ax.plot(tt, fold(tt, *p), "r-", label="FOLD")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("BOD (mg/L)")
    ax.set_title(str(name))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
    fig.savefig(os.path.join(out, f"{safe}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
