"""
benchmark.py  –  LBM Performance Benchmarking Script
SE3082 Parallel Computing – Assignment 03

Runs each implementation with varying parameters, collects timing,
and generates all performance graphs required for the report.

Usage:
    python benchmark.py --serial              # serial baseline only
    python benchmark.py --openmp              # OpenMP thread sweep
    python benchmark.py --mpi                 # MPI process sweep (once lbm_mpi is built)
    python benchmark.py --cuda                # CUDA block-size sweep (once lbm_cuda is built)
    python benchmark.py --compare             # combined bar chart from saved results
    python benchmark.py --all                 # everything available
    python benchmark.py --plot-only           # re-plot without re-running
    python benchmark.py --openmp --runs 3     # repeat each config 3 times (use median)
"""

import subprocess, os, re, json, argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("plots")

EXE_SERIAL = "./lbm_serial.exe"   if os.name == "nt" else "./lbm_serial"
EXE_OPENMP = "./lbm_openmp.exe"   if os.name == "nt" else "./lbm_openmp"
EXE_MPI    = "./lbm_mpi.exe"      if os.name == "nt" else "./lbm_mpi"
EXE_CUDA   = "./lbm_cuda.exe"     if os.name == "nt" else "./lbm_cuda"

# ── sweep parameters ─────────────────────────────────────────────────────────
THREAD_COUNTS  = [1, 2, 4, 8, 16]
PROCESS_COUNTS = [1, 2, 4, 8, 16]
CUDA_BLOCKS    = [(8, 8), (16, 16), (32, 8), (32, 16), (64, 4)]

NX, NY, NSTEPS = 600, 200, 10000    # must match C #defines

# ── helpers ──────────────────────────────────────────────────────────────────
def parse_output(text):
    """Extract elapsed time and MLUPS from program stdout."""
    m_elapsed = re.search(r"Elapsed:\s+([\d.]+)\s+s", text)
    m_mlups   = re.search(r"Throughput:\s+([\d.]+)\s+MLUPS", text)
    result = {}
    if m_elapsed: result["elapsed"] = float(m_elapsed.group(1))
    if m_mlups:   result["mlups"]   = float(m_mlups.group(1))
    return result


def run_once(cmd, env=None, label=""):
    """Run command, return parsed timing dict or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env=env, timeout=600)
        if r.returncode != 0:
            print(f"    [FAIL] {label}  stderr: {r.stderr[:200]}")
            return None
        parsed = parse_output(r.stdout)
        if "elapsed" not in parsed:
            print(f"    [WARN] could not parse timing from: {r.stdout[-200:]}")
            return None
        return parsed
    except FileNotFoundError:
        print(f"    [SKIP] executable not found: {cmd[0]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {label}")
        return None


def median_runs(cmd, env, n, label):
    """Run n times and return median elapsed, plus all values."""
    timings = []
    for i in range(n):
        r = run_once(cmd, env, label=f"{label} run {i+1}/{n}")
        if r:
            timings.append(r["elapsed"])
    if not timings:
        return None
    elapsed = float(np.median(timings))
    mlups   = (NX * NY * NSTEPS) / elapsed / 1e6
    return {"elapsed": elapsed, "elapsed_all": timings, "mlups": mlups}


def save_results(data, name):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}_results.json"
    serializable = {str(k): v for k, v in data.items()} \
                   if isinstance(data, dict) else data
    path.write_text(json.dumps(serializable, indent=2))
    print(f"  Results saved → {path}")


def load_results(name):
    path = RESULTS_DIR / f"{name}_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    try:
        return {int(k): v for k, v in data.items()}
    except (ValueError, AttributeError):
        return data


# ── runners ──────────────────────────────────────────────────────────────────
def run_serial(nruns):
    print("Running serial baseline …")
    timings = []
    for i in range(nruns):
        r = run_once([EXE_SERIAL], label=f"serial run {i+1}/{nruns}")
        if r:
            timings.append(r["elapsed"])
    if not timings:
        print("  Serial run failed.")
        return None
    elapsed = float(np.median(timings))
    result  = {"elapsed": elapsed, "elapsed_all": timings,
               "mlups": (NX * NY * NSTEPS) / elapsed / 1e6}
    print(f"  Serial → {elapsed:.3f} s  ({result['mlups']:.2f} MLUPS)")
    return result


def run_openmp(nruns):
    print("Running OpenMP thread sweep …")
    results = {}
    for n in THREAD_COUNTS:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(n)
        print(f"  threads={n:2d} …", end="", flush=True)
        r = median_runs([EXE_OPENMP], env, nruns, f"openmp-{n}")
        if r:
            results[n] = r
            print(f" {r['elapsed']:.3f} s  ({r['mlups']:.2f} MLUPS)")
        else:
            print(" FAILED")
    return results or None


def run_mpi(nruns):
    print("Running MPI process sweep …")
    results = {}
    for n in PROCESS_COUNTS:
        cmd = ["mpirun", "-n", str(n), EXE_MPI]
        print(f"  procs={n:2d} …", end="", flush=True)
        r = median_runs(cmd, None, nruns, f"mpi-{n}")
        if r:
            results[n] = r
            print(f" {r['elapsed']:.3f} s  ({r['mlups']:.2f} MLUPS)")
        else:
            print(" FAILED")
    return results or None


def run_cuda(nruns):
    print("Running CUDA block-size sweep …")
    results = {}
    for bx, by in CUDA_BLOCKS:
        label = f"{bx}x{by}"
        cmd   = [EXE_CUDA, str(bx), str(by)]
        print(f"  block={label} …", end="", flush=True)
        r = median_runs(cmd, None, nruns, f"cuda-{label}")
        if r:
            results[label] = r
            print(f" {r['elapsed']:.3f} s  ({r['mlups']:.2f} MLUPS)")
        else:
            print(" FAILED")
    return results or None


# ── plotting ─────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}

def _time_speedup_fig(x_vals, elapsed, x_label, title_prefix):
    """Return a (fig, ax1, ax2) with time and speedup sub-plots."""
    ref    = elapsed[0]
    speeds = [ref / t for t in elapsed]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"{title_prefix}  –  600×200 lattice, {NSTEPS:,} steps",
            fontsize=13, fontweight="bold"
        )

        ax1.plot(x_vals, elapsed, "o-", color="#2575B9", lw=2, ms=7)
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel("Execution Time (s)", fontsize=11)
        ax1.set_title("Execution Time", fontsize=11)
        ax1.set_xticks(x_vals)

        ax2.plot(x_vals, speeds, "o-", color="#E07B2A", lw=2, ms=7,
                 label="Measured")
        ax2.plot(x_vals, list(range(1, len(x_vals)+1)), "--",
                 color="#888", alpha=0.6, label="Ideal linear")
        ax2.set_xlabel(x_label, fontsize=11)
        ax2.set_ylabel("Speedup", fontsize=11)
        ax2.set_title("Speedup", fontsize=11)
        ax2.set_xticks(x_vals)
        ax2.legend(fontsize=10)

        plt.tight_layout()
    return fig


def plot_openmp(results):
    x_vals  = [n for n in THREAD_COUNTS if n in results]
    elapsed = [results[n]["elapsed"] for n in x_vals]
    if not x_vals:
        return
    fig = _time_speedup_fig(x_vals, elapsed, "Number of Threads", "OpenMP")
    out = PLOTS_DIR / "openmp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_mpi(results):
    x_vals  = [n for n in PROCESS_COUNTS if n in results]
    elapsed = [results[n]["elapsed"] for n in x_vals]
    if not x_vals:
        return
    fig = _time_speedup_fig(x_vals, elapsed, "Number of Processes", "MPI")
    out = PLOTS_DIR / "mpi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_cuda(results):
    labels  = list(results.keys())
    elapsed = [results[k]["elapsed"] for k in labels]
    if not labels:
        return
    ref    = elapsed[0]
    speeds = [ref / t for t in elapsed]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"CUDA Block-size Sweep  –  600×200 lattice, {NSTEPS:,} steps",
            fontsize=13, fontweight="bold"
        )

        x = range(len(labels))
        ax1.bar(x, elapsed, color="#2575B9", edgecolor="white")
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30)
        ax1.set_xlabel("Block size (x×y threads)", fontsize=11)
        ax1.set_ylabel("Execution Time (s)", fontsize=11)
        ax1.set_title("Execution Time", fontsize=11)

        ax2.bar(x, speeds, color="#E07B2A", edgecolor="white")
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30)
        ax2.set_xlabel("Block size (x×y threads)", fontsize=11)
        ax2.set_ylabel("Speedup vs first config", fontsize=11)
        ax2.set_title("Speedup", fontsize=11)

        plt.tight_layout()

    out = PLOTS_DIR / "cuda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_combined(serial_r, openmp_r, mpi_r, cuda_r):
    labels, mlups_vals = [], []

    if serial_r:
        labels.append("Serial")
        mlups_vals.append(serial_r["mlups"])

    if openmp_r:
        best = max(openmp_r, key=lambda k: openmp_r[k]["mlups"])
        labels.append(f"OpenMP\n({best} threads)")
        mlups_vals.append(openmp_r[best]["mlups"])

    if mpi_r:
        best = max(mpi_r, key=lambda k: mpi_r[k]["mlups"])
        labels.append(f"MPI\n({best} procs)")
        mlups_vals.append(mpi_r[best]["mlups"])

    if cuda_r:
        best = max(cuda_r, key=lambda k: cuda_r[k]["mlups"])
        labels.append(f"CUDA\n({best})")
        mlups_vals.append(cuda_r[best]["mlups"])

    if len(labels) < 2:
        print("  Need at least 2 implementations for combined plot — skipping.")
        return

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, mlups_vals,
                      color=colors[:len(labels)],
                      edgecolor="white", linewidth=1.2)
        ax.bar_label(bars, fmt="%.1f MLUPS", padding=5, fontsize=10)
        ax.set_ylabel("Throughput (MLUPS)", fontsize=11)
        ax.set_title(
            "LBM Implementation Comparison — Best Throughput per Platform",
            fontsize=12, fontweight="bold"
        )
        ax.set_ylim(0, max(mlups_vals) * 1.25)
        plt.tight_layout()

    out = PLOTS_DIR / "combined_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="LBM benchmark — runs implementations and plots results"
    )
    ap.add_argument("--serial",    action="store_true")
    ap.add_argument("--openmp",    action="store_true")
    ap.add_argument("--mpi",       action="store_true")
    ap.add_argument("--cuda",      action="store_true")
    ap.add_argument("--compare",   action="store_true",
                    help="Combined bar chart from saved results")
    ap.add_argument("--all",       action="store_true",
                    help="Run everything that has a built executable")
    ap.add_argument("--plot-only", action="store_true",
                    help="Re-plot from saved JSON results, no re-running")
    ap.add_argument("--runs", type=int, default=1,
                    help="Repeated runs per config; median is used (default 1)")
    args = ap.parse_args()

    if not any(vars(args).values()):
        ap.print_help()
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    nruns = args.runs

    serial_r = openmp_r = mpi_r = cuda_r = None

    if args.plot_only:
        serial_r = load_results("serial")
        openmp_r = load_results("openmp")
        mpi_r    = load_results("mpi")
        cuda_r   = load_results("cuda")
    else:
        if args.serial or args.all:
            serial_r = run_serial(nruns)
            if serial_r:
                save_results({"baseline": serial_r}, "serial")

        if args.openmp or args.all:
            openmp_r = run_openmp(nruns)
            if openmp_r:
                save_results(openmp_r, "openmp")
                print("Generating OpenMP plots …")
                plot_openmp(openmp_r)

        if args.mpi or args.all:
            mpi_r = run_mpi(nruns)
            if mpi_r:
                save_results(mpi_r, "mpi")
                print("Generating MPI plots …")
                plot_mpi(mpi_r)

        if args.cuda or args.all:
            cuda_r = run_cuda(nruns)
            if cuda_r:
                save_results(cuda_r, "cuda")
                print("Generating CUDA plots …")
                plot_cuda(cuda_r)

    if args.compare or args.all or args.plot_only:
        if serial_r is None:
            loaded = load_results("serial")
            if loaded and "baseline" in loaded:
                serial_r = loaded["baseline"]
        if openmp_r is None: openmp_r = load_results("openmp")
        if mpi_r    is None: mpi_r    = load_results("mpi")
        if cuda_r   is None: cuda_r   = load_results("cuda")
        print("Generating combined comparison plot …")
        plot_combined(serial_r, openmp_r, mpi_r, cuda_r)

    print("\nAll done.")


if __name__ == "__main__":
    main()
