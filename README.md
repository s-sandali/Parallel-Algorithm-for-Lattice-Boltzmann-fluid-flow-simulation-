# SE3082 – Parallel Computing Assignment 03  
## 2D Lattice Boltzmann Method (D2Q9-BGK) for Incompressible Fluid Flow Past a Cylinder

**Course:** SE3082 – Parallel Computing  
**Student:** Sandali Sandagomi  

---

## Overview

This repository contains the **serial baseline** implementation of a 2D Lattice Boltzmann Method (LBM) fluid simulation. The simulation models incompressible flow past a circular cylinder and, at the chosen Reynolds number (~90), produces the classic **Kármán vortex street** — an alternating pattern of vortices shed from either side of the cylinder.

The serial implementation is Phase 1 of the assignment. Subsequent phases will parallelise the same algorithm using **OpenMP**, **MPI**, and **CUDA**, with the serial output serving as the correctness reference.

---

## Background

The Lattice Boltzmann Method is a mesoscopic Computational Fluid Dynamics (CFD) technique. Rather than solving the Navier-Stokes equations directly, it evolves **particle distribution functions** on a discrete lattice. The Chapman-Enskog expansion proves that the scheme recovers incompressible Navier-Stokes in the macroscopic limit.

**D2Q9 model** — 2 dimensions, 9 discrete velocities per cell:

```
  6   2   5
    \ | /
  3 - 0 - 1
    / | \
  7   4   8
```

Each timestep has two stages:
- **Collision (BGK):** relax local distributions toward equilibrium using the single-relaxation-time operator (τ = 0.6).
- **Streaming:** propagate each distribution to the neighbouring cell along its velocity direction.

Solid boundaries (cylinder surface, top/bottom walls) use **bounce-back**: incoming distributions are reversed, enforcing a no-slip condition.

---

## Simulation Parameters

| Parameter | Value | Notes |
|---|---|---|
| Lattice size | 600 × 200 | x × y cells |
| Timesteps | 10 000 | |
| Relaxation time τ | 0.6 | ν = (τ − 0.5)/3 ≈ 0.0333 |
| Inlet velocity U | 0.1 | lattice units |
| Cylinder centre | (150, 100) | (NX/4, NY/2) |
| Cylinder radius | 15 | cells |
| Reynolds number | ~90 | vortex-shedding regime |
| Output interval | every 100 steps | 100 PGM frames total |

Re = U·D / ν = 0.1 × 30 / 0.0333 ≈ 90, which sits comfortably in the Kármán vortex street regime.

---

## Algorithm Steps (per timestep)

1. **Macroscopic variables** — compute density ρ and velocity (u_x, u_y) as moments of f at every fluid cell.
2. **BGK collision** — relax f toward f_eq with rate 1/τ at every fluid cell.
3. **Pull-style streaming with bounce-back** — each fluid cell pulls f_i from its upstream neighbour; if that neighbour is solid, bounce-back (f_opp from self) is used instead.
4. **Zou-He inlet** (x = 0) — prescribe u = (U_INLET, 0) and recompute the three unknown incoming distributions consistently.
5. **Zero-gradient outlet** (x = NX−1) — copy distributions from the penultimate column.
6. **Buffer swap** — f and f_new are swapped; no data is copied.
7. **Output** — velocity magnitude field saved as a PGM image every 100 steps.

---

## File Structure

```
assignment3/
├── lbm_serial.c        # Serial C implementation (D2Q9-BGK LBM)
├── lbm_serial.exe      # Pre-compiled Windows binary
├── frame_00100.pgm     # Velocity magnitude frame at step 100
├── frame_00200.pgm     # ...
│   ...
└── frame_10000.pgm     # Final frame (step 10 000)
```

> The 100 PGM frames record the evolution of the velocity magnitude field from early transient to fully developed vortex shedding.

---

## Build and Run

### Prerequisites

- GCC (or any C99-compatible compiler)
- `libm` (math library — standard on Linux/macOS; included on Windows with MinGW)

### Compile

```bash
gcc -O3 -o lbm_serial lbm_serial.c -lm
```

### Run

```bash
./lbm_serial
```

Expected console output:

```
D2Q9-BGK LBM serial: 600x200 lattice, 10000 steps, tau=0.60, U=0.100
Reynolds number ~ 90.0 (based on cylinder diameter)
  step   100 / 10000  saved frame
  step   200 / 10000  saved frame
  ...
Done.
Elapsed:    XX.XXX s
Throughput: XX.XX MLUPS (mega lattice updates per second)
```

### Visualise output

**View a single frame** with any PGM-capable viewer (e.g. GIMP, IrfanView, `eog` on Linux).

**Convert frames to an animated GIF** (requires ImageMagick):

```bash
convert -delay 5 frames/frame_*.pgm output.gif
```

In the output images, pixel brightness encodes velocity magnitude — brighter regions are faster flow. The cylinder appears black (zero velocity, obstacle). The alternating vortices shed behind the cylinder become clearly visible from around step 2000 onward.

---

## Performance

On a typical modern desktop CPU (single core), the 600×200 × 10 000-step run completes in roughly **30–60 seconds**, achieving approximately **20–40 MLUPS** (mega lattice updates per second). Exact figures depend on CPU and compiler.

---

## Memory Layout

Distribution functions are stored as a flat 1D array with index scheme:

```
f[x][y][i]  →  flat index = (x * NY + y) * 9 + i
```

The direction index `i` is the fastest-varying dimension, which keeps the nine distributions for a single cell contiguous in memory. This layout favours cache efficiency during the inner loop over directions in both the collision and macroscopic steps.

---

## References

1. P. L. Bhatnagar, E. P. Gross, and M. Krook, "A model for collision processes in gases," *Physical Review*, vol. 94, no. 3, pp. 511–525, 1954.
2. P. Mocz, "Create Your Own Lattice Boltzmann Simulation (With Python)," Princeton University, 2020. [GitHub](https://github.com/pmocz/latticeboltzmann-python) — studied as algorithmic reference.
3. Palabos Project, "Lattice Boltzmann sample codes," University of Geneva. [palabos.unige.ch](https://palabos.unige.ch/get-started/lattice-boltzmann/lattice-boltzmann-sample-codes-various-other-programming-languages) — C reference studied.
4. T. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*, Springer, 2017.

---

## Notes

- The small sinusoidal perturbation in the initial velocity field (`0.02 * cos(...)`) breaks perfect left-right symmetry so that vortex shedding develops naturally within the simulation window without requiring an artificially long transient.
- Top and bottom rows are treated as solid walls (no-slip), confining the flow in a channel of height NY − 2 fluid cells.
- The outlet uses a simple zero-gradient condition rather than a more expensive non-reflecting boundary; this is sufficient for this Reynolds number and domain length.
