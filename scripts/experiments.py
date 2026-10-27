#!/usr/bin/env python3
"""Run every experiment in the README and write summary tables to results/.

    python3 scripts/experiments.py [--build build] [--raw raw] [--only NAME ...]

Experiments
  validation   Metropolis on a 128x128 lattice vs Onsager's exact solution
  binder       Wolff runs near T_c; Binder cumulant crossing locates T_c
  autocorr     integrated autocorrelation time at T_c, Metropolis vs Wolff
  kernels      single-core throughput of the three Metropolis kernels
  scaling      OpenMP and MPI throughput on this machine

Raw time series go to --raw (not committed); summaries go to results/.
"""
import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys

import numpy as np
from scipy.special import ellipk

TC = 2.0 / math.log(1.0 + math.sqrt(2.0))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


# ---------------------------------------------------------------- exact results
def onsager_m(T):
    """Spontaneous magnetisation of the infinite lattice (Onsager / Yang)."""
    if T >= TC:
        return 0.0
    return (1.0 - math.sinh(2.0 / T) ** -4) ** 0.125


def onsager_e(T):
    """Energy per site of the infinite lattice (Onsager)."""
    if abs(T - TC) < 1e-12:
        return -math.sqrt(2.0)  # the elliptic-integral term vanishes at T_c
    b = 1.0 / T
    k = 2.0 * math.sinh(2 * b) / math.cosh(2 * b) ** 2
    K = ellipk(k * k)  # scipy takes the parameter m = k^2
    return -1.0 / math.tanh(2 * b) * (1.0 + 2.0 / math.pi * (2.0 * math.tanh(2 * b) ** 2 - 1.0) * K)


# ---------------------------------------------------------------- statistics
def tau_int(x, c=6.0):
    """Integrated autocorrelation time with Sokal's automatic window."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    for w in range(1, n):
        tau += acf[w]
        if w >= c * tau:
            return tau, w
    return tau, n - 1


def tau_err(tau, window, n):
    """Madras-Sokal error estimate for tau_int."""
    return tau * math.sqrt(2.0 * (2 * window + 1) / n)


def mean_err(x):
    """Mean and its standard error, inflated by 2 * tau_int for correlated data."""
    x = np.asarray(x, dtype=float)
    tau, _ = tau_int(x)
    return x.mean(), math.sqrt(x.var(ddof=1) * max(2.0 * tau, 1.0) / len(x))


def binder_jackknife(m, nblocks=50):
    """Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2) with a blocked jackknife error."""
    m = np.asarray(m, dtype=float)
    n = len(m) // nblocks * nblocks
    m2 = (m[:n] ** 2).reshape(nblocks, -1).mean(axis=1)
    m4 = (m[:n] ** 4).reshape(nblocks, -1).mean(axis=1)
    u = lambda a2, a4: 1.0 - a4 / (3.0 * a2 * a2)
    full = u(m2.mean(), m4.mean())
    jk = np.array([u(np.delete(m2, i).mean(), np.delete(m4, i).mean()) for i in range(nblocks)])
    err = math.sqrt((nblocks - 1) / nblocks * ((jk - jk.mean()) ** 2).sum())
    return full, err


# ---------------------------------------------------------------- running binaries
def run(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=e).stdout.strip()
    return dict(kv.split("=", 1) for kv in out.split())


def read_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data


def write_csv(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote results/{name}")


# ---------------------------------------------------------------- experiments
def validation(a):
    L, sweeps, therm, every = 128, 50000, 5000, 10
    temps = [round(1.6 + 0.1 * i, 2) for i in range(19)]
    rows = []
    for T in temps:
        path = os.path.join(a.raw, f"val_L{L}_T{T}.csv")
        run([f"{a.build}/ising", "--L", str(L), "--T", str(T), "--sweeps", str(sweeps), "--therm", str(therm),
             "--every", str(every), "--seed", "101", "--out", path,
             # Below T_c a random start can freeze into long-lived stripe domains on a
             # periodic lattice, so the ordered phase is started from all spins up.
             "--start", "cold" if T < TC else "hot"], {"OMP_NUM_THREADS": str(a.threads)})
        d = read_csv(path)
        e, de = mean_err(d["e"])
        am, dam = mean_err(np.abs(d["m"]))
        rows.append([T, f"{e:.6f}", f"{de:.6f}", f"{am:.6f}", f"{dam:.6f}", f"{onsager_e(T):.6f}", f"{onsager_m(T):.6f}"])
        print(f"  T={T:.2f}  e={e:.5f}+-{de:.5f} (exact {onsager_e(T):.5f})  |m|={am:.5f}+-{dam:.5f} (exact {onsager_m(T):.5f})")
    write_csv(f"validation_metropolis_L{L}.csv", ["T", "e", "e_err", "abs_m", "abs_m_err", "e_onsager", "m_onsager"], rows)


def binder(a):
    sizes = [16, 32, 64, 128]
    temps = [round(2.22 + 0.01 * i, 2) for i in range(11)]
    rows = []
    for L in sizes:
        for T in temps:
            path = os.path.join(a.raw, f"binder_L{L}_T{T}.csv")
            run([f"{a.build}/wolff", "--L", str(L), "--T", str(T), "--clusters", "200000", "--therm", "5000",
                 "--every", "2", "--seed", str(1000 + L), "--out", path])
            d = read_csv(path)
            u, du = binder_jackknife(d["m"])
            rows.append([L, T, f"{u:.6f}", f"{du:.6f}"])
        print(f"  L={L} done")
    write_csv("binder_wolff.csv", ["L", "T", "U4", "U4_err"], rows)


def autocorr(a):
    rows = []
    for L in [8, 16, 32, 64, 128]:
        sweeps = max(200000, int(1500 * 39 * (L / 32) ** 2.17))  # ~1500 tau from the L=32 pilot
        path = os.path.join(a.raw, f"ac_met_L{L}.csv")
        run([f"{a.build}/ising", "--L", str(L), "--sweeps", str(sweeps), "--therm", str(sweeps // 20), "--every", "1",
             "--seed", "7", "--out", path], {"OMP_NUM_THREADS": str(a.threads)})
        d = read_csv(path)
        x = d["m"] ** 2
        tau, w = tau_int(x)
        rows.append([L, "metropolis", f"{tau:.3f}", f"{tau_err(tau, w, len(x)):.3f}", len(x)])
        print(f"  metropolis L={L}: tau={tau:.2f} sweeps ({len(x)} samples)")

        path = os.path.join(a.raw, f"ac_wolff_L{L}.csv")
        run([f"{a.build}/wolff", "--L", str(L), "--clusters", "200000", "--therm", "5000", "--seed", "7", "--out", path])
        d = read_csv(path)
        x = d["m"] ** 2
        tau_c, w = tau_int(x)
        per = d["sweeps"][-1] / len(x)  # sweep-equivalents per cluster
        rows.append([L, "wolff", f"{tau_c * per:.4f}", f"{tau_err(tau_c, w, len(x)) * per:.4f}", len(x)])
        print(f"  wolff      L={L}: tau={tau_c * per:.3f} sweeps ({tau_c:.2f} clusters)")
    write_csv("autocorr_Tc.csv", ["L", "algorithm", "tau_int_sweeps", "tau_err", "samples"], rows)


def kernels(a):
    rows = []
    for L in [256, 512, 1024, 2048]:
        for k in ["baseline", "blocked", "batched"]:
            sweeps = max(20, int(2.0e9 * (0.06 if k == "baseline" else 0.2) / L ** 2 / 1.0))
            vals = []
            for rep in range(3):
                r = run([f"{a.build}/ising", "--L", str(L), "--sweeps", str(sweeps), "--therm", "0", "--every", "0",
                         "--kernel", k, "--seed", str(rep)], {"OMP_NUM_THREADS": "1"})
                vals.append(float(r["flips_per_ns"]))
            rows.append([L, k, f"{max(vals):.4f}", f"{statistics.median(vals):.4f}"])
            print(f"  L={L} {k:8s} best {max(vals):.4f} flips/ns")
    write_csv("bench_kernels.csv", ["L", "kernel", "best_flips_per_ns", "median_flips_per_ns"], rows)


def scaling(a):
    L, sweeps = 2048, 120
    cores = os.cpu_count()
    rows = []
    for t in range(1, cores + 1):
        vals = [float(run([f"{a.build}/ising", "--L", str(L), "--sweeps", str(sweeps), "--therm", "0", "--every", "0"],
                          {"OMP_NUM_THREADS": str(t)})["flips_per_ns"]) for _ in range(3)]
        rows.append(["openmp", t, L, f"{max(vals):.4f}"])
        print(f"  openmp threads={t}: {max(vals):.4f}")
    mpirun = ["mpirun", "--oversubscribe", "--allow-run-as-root", "--bind-to", "core"]
    for r in range(1, cores + 1):
        vals = [float(run(mpirun + ["-np", str(r), f"{a.build}/ising_mpi", "--L", str(L), "--sweeps", str(sweeps),
                                    "--therm", "0", "--every", "0"], {"OMP_NUM_THREADS": "1"})["flips_per_ns"])
                for _ in range(3)]
        rows.append(["mpi", r, L, f"{max(vals):.4f}"])
        print(f"  mpi ranks={r}: {max(vals):.4f}")
    write_csv("bench_scaling.csv", ["backend", "workers", "L", "flips_per_ns"], rows)


EXPERIMENTS = {"validation": validation, "binder": binder, "autocorr": autocorr, "kernels": kernels, "scaling": scaling}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", default=os.path.join(ROOT, "build"))
    p.add_argument("--raw", default=os.path.join(ROOT, "raw"))
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="OpenMP threads for physics runs")
    p.add_argument("--only", nargs="*", choices=list(EXPERIMENTS))
    a = p.parse_args()
    os.makedirs(a.raw, exist_ok=True)
    for name in a.only or EXPERIMENTS:
        print(f"[{name}]")
        EXPERIMENTS[name](a)
