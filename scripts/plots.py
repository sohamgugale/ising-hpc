#!/usr/bin/env python3
"""Make the README figures from the summary tables in results/."""
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TC = 2.0 / math.log(1.0 + math.sqrt(2.0))

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RES, name), dpi=110)
    plt.close(fig)
    print(f"wrote results/{name}")


def validation():
    rows = load("validation_metropolis_L128.csv")
    T = np.array([float(r["T"]) for r in rows])
    fine = np.linspace(1.5, 3.5, 400)
    from experiments import onsager_e, onsager_m  # noqa: E402

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(fine, [onsager_m(t) for t in fine], "k-", lw=1, label="Onsager (L = ∞)")
    ax[0].errorbar(T, [float(r["abs_m"]) for r in rows], [float(r["abs_m_err"]) for r in rows], fmt="o", ms=3,
                   color="C0", label="Metropolis, L = 128")
    ax[0].set(xlabel="T", ylabel="⟨|m|⟩", title="Magnetisation")
    ax[1].plot(fine, [onsager_e(t) for t in fine], "k-", lw=1, label="Onsager (L = ∞)")
    ax[1].errorbar(T, [float(r["e"]) for r in rows], [float(r["e_err"]) for r in rows], fmt="o", ms=3, color="C1",
                   label="Metropolis, L = 128")
    ax[1].set(xlabel="T", ylabel="⟨e⟩", title="Energy per site")
    for a in ax:
        a.axvline(TC, color="gray", ls=":", lw=1)
        a.legend(fontsize=7)
    save(fig, "validation.png")


def binder():
    rows = load("binder_wolff.csv")
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    for i, L in enumerate(sorted({int(r["L"]) for r in rows})):
        sub = [r for r in rows if int(r["L"]) == L]
        ax.errorbar([float(r["T"]) for r in sub], [float(r["U4"]) for r in sub], [float(r["U4_err"]) for r in sub],
                    fmt="o-", ms=3, lw=1, color=f"C{i}", label=f"L = {L}")
    ax.axvline(TC, color="gray", ls=":", lw=1, label="exact T_c")
    ax.set(xlabel="T", ylabel="Binder cumulant U₄", title="Finite-size crossing (Wolff)")
    ax.legend(fontsize=7)
    save(fig, "binder.png")


def autocorr():
    rows = load("autocorr_Tc.csv")
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    for i, alg in enumerate(["metropolis", "wolff"]):
        sub = [r for r in rows if r["algorithm"] == alg]
        L = np.array([int(r["L"]) for r in sub])
        tau = np.array([float(r["tau_int_sweeps"]) for r in sub])
        err = np.array([float(r["tau_err"]) for r in sub])
        ax.errorbar(L, tau, err, fmt="o", ms=4, color=f"C{i}")
        fit = L >= 16
        z, c = np.polyfit(np.log(L[fit]), np.log(tau[fit]), 1)
        xs = np.array([L.min(), L.max()])
        ax.plot(xs, np.exp(c) * xs ** z, "-", lw=1, color=f"C{i}", label=f"{alg.capitalize()}: z ≈ {z:.2f}")
    ax.set(xscale="log", yscale="log", xlabel="L", ylabel="τ_int of m² (sweeps)", title="Critical slowing down at T_c")
    ax.legend(fontsize=7)
    save(fig, "autocorr.png")


def kernels():
    rows = load("bench_kernels.csv")
    Ls = sorted({int(r["L"]) for r in rows})
    ks = ["baseline", "blocked", "batched"]
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    w = 0.26
    for i, k in enumerate(ks):
        vals = [float(next(r for r in rows if int(r["L"]) == L and r["kernel"] == k)["best_flips_per_ns"]) for L in Ls]
        ax.bar(np.arange(len(Ls)) + (i - 1) * w, vals, w, label=k, color=f"C{i}")
    ax.set_xticks(np.arange(len(Ls)), [str(L) for L in Ls])
    ax.set(xlabel="L", ylabel="spin flips / ns (1 core)", title="Kernel optimisation", ylim=(0, 0.3))
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    save(fig, "kernels.png")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for f in (validation, binder, autocorr, kernels):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: {e}")
