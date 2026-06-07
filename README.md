# SE3082 – Parallel Computing Assignment 03
## 2D Lattice Boltzmann Method (D2Q9-BGK) for Incompressible Fluid Flow Past a Cylinder

**Course:** SE3082 – Parallel Computing  
**Student:** Sandali Sandagomi

---

## Overview

Four implementations of a 2D D2Q9-BGK Lattice Boltzmann Method (LBM) fluid simulation, parallelised using three different paradigms:

| File | Parallelism | Tool |
|---|---|---|
| `lbm_serial.c` | None (baseline) | GCC |
| `lbm_openmp.c` | Shared memory | OpenMP |
| `lbm_mpi.c` | Distributed memory | MPI |
| `lbm_cuda.cu` | GPU | CUDA |

The simulation models 2D incompressible flow past a circular cylinder. At Reynolds number ~90, the flow produces a **Kármán vortex street** — an alternating pattern of vortices shed from either side of the cylinder. All four implementations produce physically identical results and serve as the basis for comparative performance evaluation.

---

## Background

The Lattice Boltzmann Method is a mesoscopic CFD technique that evolves **particle distribution functions** on a discrete lattice rather than solving the Navier-Stokes equations directly. The Chapman-Enskog expansion proves the scheme recovers incompressible Navier-Stokes in the macroscopic limit.

**D2Q9 model** — 2 dimensions, 9 discrete velocities per cell:

```
  6   2   5
    \ | /
  3 - 0 - 1
    / | \
  7   4   8
```

Each timestep:
1. **Macroscopic step** — compute density ρ and velocity (u_x, u_y) as moments of f.
2. **BGK collision** — relax f toward equilibrium with rate 1/τ.
3. **Pull-style streaming + bounce-back** — propagate f to neighbours; solid cells reverse the distribution (no-slip).
4. **Zou-He inlet** (x = 0) — prescribe u = (U_INLET, 0).
5. **Zero-gradient outlet** (x = NX−1) — copy from penultimate column.
6. **Buffer swap** — pointer swap, no data copy.

---

## Simulation Parameters

| Parameter | Value | Notes |
|---|---|---|
| Lattice size | 600 × 200 | x × y cells |
| Timesteps | 10 000 | |
| Relaxation time τ | 0.6 | ν = (τ − 0.5)/3 ≈ 0.0333 |
| Inlet velocity U | 0.1 | lattice units |
| Cylinder centre | (150, 100) | NX/4, NY/2 |
| Cylinder radius | 15 | cells |
| Reynolds number | ~90 | Kármán vortex shedding regime |
| Output interval | every 100 steps | 100 frames total |

---

## File Structure

```
assignment3/
├── lbm_serial.c          # Serial baseline
├── lbm_openmp.c          # OpenMP shared-memory parallelisation
├── lbm_mpi.c             # MPI distributed-memory parallelisation
├── lbm_cuda.cu           # CUDA GPU parallelisation
│
├── benchmark.py          # Runs all implementations, collects timing, plots graphs
├── visualize.py          # Makes animated GIFs and comparison snapshots
│
├── frames/               # Generated at runtime (git-ignored)
│   ├── serial/           # Grayscale PGM frames — serial
│   ├── serial_ppm/       # Color vorticity PPM frames — serial
│   ├── openmp/           # Grayscale PGM frames — OpenMP
│   ├── openmp_ppm/       # Color vorticity PPM frames — OpenMP
│   ├── mpi/              # Grayscale PGM frames — MPI
│   ├── mpi_ppm/          # Color vorticity PPM frames — MPI
│   ├── cuda/             # Grayscale PGM frames — CUDA
│   └── cuda_ppm/         # Color vorticity PPM frames — CUDA
│
├── animations/           # GIF outputs from visualize.py (git-ignored)
├── plots/                # Performance graphs from benchmark.py (git-ignored)
├── results/              # JSON timing data from benchmark.py (git-ignored)
│
└── README.md
```

Each implementation writes **two sets of frames** per run:
- **PGM** (grayscale) — velocity magnitude, used for correctness comparison
- **PPM** (color) — vorticity field (red = positive, white = zero, blue = negative), best for visualization

---

## Prerequisites

| Tool | Used for |
|---|---|
| GCC | Compile serial and OpenMP |
| MS-MPI (Windows) or OpenMPI (Linux) | Compile and run MPI |
| CUDA Toolkit + NVIDIA GPU | Compile and run CUDA |
| Python 3 + `numpy`, `matplotlib`, `Pillow` | benchmark.py and visualize.py |

---

## Build

### Serial
```powershell
gcc -O3 -o lbm_serial lbm_serial.c -lm
```

### OpenMP
```powershell
gcc -O3 -fopenmp -o lbm_openmp lbm_openmp.c -lm
```

### MPI (Windows — MS-MPI)
Install MS-MPI runtime (`msmpisetup.exe`) and SDK (`msmpisdk.msi`) from the [Microsoft MPI releases page](https://github.com/microsoft/Microsoft-MPI/releases), then:
```powershell
gcc -O3 -o lbm_mpi lbm_mpi.c -lm `
  -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" `
  -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" `
  -lmsmpi
```

### MPI (Linux / WSL)
```bash
mpicc -O3 -o lbm_mpi lbm_mpi.c -lm
```

### CUDA
```bash
nvcc -O3 -o lbm_cuda lbm_cuda.cu -lm
```

---

## Run

### Serial
```powershell
./lbm_serial
```

### OpenMP — set thread count via environment variable
```powershell
$env:OMP_NUM_THREADS=1;  ./lbm_openmp
$env:OMP_NUM_THREADS=2;  ./lbm_openmp
$env:OMP_NUM_THREADS=4;  ./lbm_openmp
$env:OMP_NUM_THREADS=8;  ./lbm_openmp
$env:OMP_NUM_THREADS=16; ./lbm_openmp
```

### MPI — vary number of processes
```powershell
mpiexec -n 1  ./lbm_mpi
mpiexec -n 2  ./lbm_mpi
mpiexec -n 4  ./lbm_mpi
mpiexec -n 8  ./lbm_mpi
mpiexec -n 16 ./lbm_mpi
```

### CUDA — default block size (16×16), or specify custom
```bash
./lbm_cuda              # default 16×16 blocks
./lbm_cuda 8  8         # 8×8 blocks
./lbm_cuda 32 8         # 32×8 blocks
./lbm_cuda 32 16        # 32×16 blocks
./lbm_cuda 64 4         # 64×4 blocks
```

---

## Console Output (per run)

Each implementation prints timing at the end. The Python benchmark script parses these lines automatically.

**Serial:**
```
D2Q9-BGK LBM serial: 600x200 lattice, 10000 steps, tau=0.60, U=0.100
Reynolds number ~ 90.0
  step   100 / 10000  saved frame
  ...
Done.
Elapsed:    XX.XXX s
Throughput: XX.XX MLUPS (mega lattice updates per second)
```

**OpenMP** (adds thread count):
```
Threads: 8
...
Elapsed:    XX.XXX s
```

**MPI** (adds process count):
```
Processes: 4
...
Elapsed:    XX.XXX s
```

**CUDA** (adds block configuration):
```
Block size: 16x16 = 256 threads/block
...
Elapsed:    XX.XXX s
```

---

## Benchmarking — `benchmark.py`

Automates running each implementation with varying parameters, collects timing, and generates all performance graphs required for the report.

### Run the serial baseline
```powershell
python3 benchmark.py --serial
```

### Run the OpenMP thread sweep (1, 2, 4, 8, 16 threads)
```powershell
python3 benchmark.py --openmp
```
Generates `plots/openmp.png` — execution time and speedup side by side.

### Run the MPI process sweep
```powershell
python3 benchmark.py --mpi
```
Generates `plots/mpi.png`.

### Run the CUDA block-size sweep
```powershell
python3 benchmark.py --cuda
```
Generates `plots/cuda.png`.

### Run everything at once
```powershell
python3 benchmark.py --all
```

### Generate combined comparison chart (from saved results)
```powershell
python3 benchmark.py --compare
```
Generates `plots/combined_comparison.png` — bar chart of best MLUPS per implementation.

### Re-plot without re-running (uses saved JSON results)
```powershell
python3 benchmark.py --plot-only
```

### Repeat each config N times and use the median (for report accuracy)
```powershell
python3 benchmark.py --openmp --runs 3
```

Timing results are saved to `results/` as JSON files after each sweep.

---

## Visualization — `visualize.py`

### Make an animated GIF (color vorticity — recommended)
```powershell
python3 visualize.py --gif serial_ppm
python3 visualize.py --gif openmp_ppm
python3 visualize.py --gif mpi_ppm
python3 visualize.py --gif cuda_ppm
```
Output goes to `animations/<folder>.gif`.

### Make a grayscale velocity magnitude GIF
```powershell
python visualize.py --gif serial
```

### Control animation speed (default 80 ms/frame ≈ 12 fps)
```powershell
python visualize.py --gif serial_ppm --delay 50    # faster (~20 fps)
python visualize.py --gif serial_ppm --delay 150   # slower (~7 fps)
```

### Side-by-side comparison snapshot (for the report)
```powershell
python3 visualize.py --compare 5000 --folders serial_ppm openmp_ppm
```
Saves a single PNG to `plots/compare_serial_ppm_vs_openmp_ppm_step05000.png`.

### Pixel-level correctness check between two implementations
```powershell
python3 visualize.py --diff serial openmp
python3 visualize.py --diff serial mpi
```
Reports max and mean pixel difference across all frames. Identical results print `✓`.

### Open a GIF
```powershell
start animations\serial_ppm.gif
```

---

## Parallelisation Strategies

### OpenMP (`lbm_openmp.c`)
- `#pragma omp parallel for collapse(2) schedule(static)` over all (x, y) cells in the macroscopic, collision, and streaming steps.
- The collision step is embarrassingly parallel (purely local).
- Pull-style streaming writes each thread to its own (x, y) location in `fnew` — no write conflicts.
- Wall time measured with `omp_get_wtime()`.

### MPI (`lbm_mpi.c`)
- 1D domain decomposition along x — the 600-column lattice is split into contiguous strips, one per process.
- Each process has two extra **ghost columns** (one left, one right) for neighbour data.
- After collision, a single `MPI_Sendrecv` call exchanges ghost columns with left and right neighbours simultaneously (avoids deadlock).
- Output: local `ux`/`uy` arrays are gathered to rank 0 via `MPI_Gatherv` for file I/O.
- Timing with `MPI_Wtime()` on rank 0.

### CUDA (`lbm_cuda.cu`)
- One CUDA thread per lattice cell; 2D thread blocks over the (x, y) grid.
- Five GPU kernels per timestep: `macroscopic_kernel`, `collision_kernel`, `streaming_kernel`, `inlet_kernel`, `outlet_kernel`.
- D2Q9 lattice constants stored in `__constant__` memory — cached, read-only, broadcast to all threads in a warp.
- All arrays remain on the GPU for the entire simulation. Data is copied back to the host only every 100 steps for output.
- Device pointer swap (`d_f ↔ d_fnew`) requires no data movement.
- Timing with `cudaEvent_t` for accurate GPU-side measurement.

---

## Memory Layout

All implementations use the same flat 1D indexing scheme for the distribution functions:

```
f[x][y][i]  →  flat index = (x * NY + y) * 9 + i
```

The direction index `i` is the fastest-varying dimension, keeping all 9 distributions for one cell contiguous in memory — cache-friendly for the inner direction loop in both collision and macroscopic steps.

---

## References

1. P. L. Bhatnagar, E. P. Gross, and M. Krook, "A model for collision processes in gases," *Physical Review*, vol. 94, no. 3, pp. 511–525, 1954.
2. P. Mocz, "Create Your Own Lattice Boltzmann Simulation (With Python)," Princeton University, 2020. [GitHub](https://github.com/pmocz/latticeboltzmann-python) — algorithmic reference.
3. Palabos Project, "Lattice Boltzmann sample codes," University of Geneva. [palabos.unige.ch](https://palabos.unige.ch/get-started/lattice-boltzmann/lattice-boltzmann-sample-codes-various-other-programming-languages) — C reference studied.
4. T. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*, Springer, 2017.
