/*
 * lbm_cuda.cu
 *
 * CUDA-parallelised D2Q9-BGK Lattice Boltzmann Method:
 * 2D incompressible flow past a circular cylinder.
 *
 * Parallelisation strategy:
 *   - The 600x200 lattice maps naturally to a 2D CUDA grid.
 *     Each thread handles one lattice cell (x, y).
 *     Block size is configurable (default 16x16 = 256 threads per block);
 *     the grid dimensions are computed to cover the full lattice.
 *   - Three main GPU kernels run each timestep:
 *       1. macroscopic_kernel  — compute rho, ux, uy from f  (purely local)
 *       2. collision_kernel    — BGK relaxation of f          (purely local)
 *       3. streaming_kernel    — pull-style streaming + bounce-back (stencil)
 *     Two small kernels handle boundary conditions:
 *       4. inlet_kernel        — Zou-He velocity inlet at x = 0
 *       5. outlet_kernel       — zero-gradient outlet at x = NX-1
 *   - D2Q9 lattice constants (ex, ey, w_, opp) are stored in CUDA
 *     __constant__ memory — read-only, cached, and broadcast to all threads.
 *   - All data arrays live on the GPU for the entire simulation.  Data is
 *     only copied back to the host every OUTPUT_INTERVAL steps for PGM output.
 *   - Timing uses CUDA events for accurate GPU-side measurement.
 *
 * Build:   nvcc -O3 -o lbm_cuda lbm_cuda.cu -lm
 * Run:     ./lbm_cuda
 *          ./lbm_cuda 32 32        (custom block size, e.g. 32x32)
 *
 * References (algorithmic only; this code is original):
 *   - Mocz (2020), Palabos LBM codes, Krüger et al. (2017).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------ */
/* Output directories (matching OpenMP structure)                     */
/* ------------------------------------------------------------------ */
#define FRAMES_DIR      "frames/cuda"
#define FRAMES_PPM_DIR  "frames/cuda_ppm"

/* ------------------------------------------------------------------ */
/* Simulation parameters                                              */
/* ------------------------------------------------------------------ */
#define NX               600
#define NY               200
#define NSTEPS           10000
#define OUTPUT_INTERVAL  100

#define U_INLET          0.1
#define TAU              0.6

#define CYL_X            (NX/4)
#define CYL_Y            (NY/2)
#define CYL_R            15

/* Default block dimensions (can be overridden via command-line args) */
#define DEFAULT_BX       16
#define DEFAULT_BY       16

/* ------------------------------------------------------------------ */
/* CUDA error-checking macro (from lecture notes)                     */
/* Kernel launches fail SILENTLY without this.                        */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d -> %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* ------------------------------------------------------------------ */
/* D2Q9 lattice constants in CUDA __constant__ memory.                */
/* __constant__ is cached, read-only memory that is broadcast to all  */
/* threads in a warp — ideal for small lookup tables like these.      */
/* ------------------------------------------------------------------ */
__constant__ int    d_ex[9];
__constant__ int    d_ey[9];
__constant__ double d_w[9];
__constant__ int    d_opp[9];

/* Host copies for initialisation and output */
static const int    h_ex[9]  = {  0,  1,  0, -1,  0,  1, -1, -1,  1 };
static const int    h_ey[9]  = {  0,  0,  1,  0, -1,  1,  1, -1, -1 };
static const double h_w[9]   = { 4.0/9.0,
                                  1.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0,
                                  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
static const int    h_opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

/* ------------------------------------------------------------------ */
/* Index helpers — __host__ __device__ so usable on both CPU and GPU  */
/* ------------------------------------------------------------------ */
__host__ __device__ static inline int idx_f(int x, int y, int i) {
    return ((x * NY) + y) * 9 + i;
}
__host__ __device__ static inline int idx_c(int x, int y) {
    return x * NY + y;
}

/* ------------------------------------------------------------------ */
/* Device: equilibrium distribution                                   */
/* ------------------------------------------------------------------ */
__device__ static inline double feq_i(int i, double rho, double ux, double uy) {
    double cu  = 3.0 * (d_ex[i]*ux + d_ey[i]*uy);
    double usq = 1.5 * (ux*ux + uy*uy);
    return d_w[i] * rho * (1.0 + cu + 0.5*cu*cu - usq);
}

/* Host version for initialisation */
static inline double feq_i_host(int i, double rho, double ux, double uy) {
    double cu  = 3.0 * (h_ex[i]*ux + h_ey[i]*uy);
    double usq = 1.5 * (ux*ux + uy*uy);
    return h_w[i] * rho * (1.0 + cu + 0.5*cu*cu - usq);
}

/* ================================================================== */
/* GPU KERNELS                                                        */
/* ================================================================== */

/* ------------------------------------------------------------------ *
 * Kernel 1: compute macroscopic variables (density, velocity).       *
 * Each thread handles one cell — purely local, no neighbour access.  *
 * ------------------------------------------------------------------ */
__global__ void macroscopic_kernel(const double *f, double *rho,
                                   double *ux, double *uy,
                                   const int *obstacle)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;   /* boundary check */

    int c = idx_c(x, y);
    double r = 0.0, mx = 0.0, my = 0.0;
    for (int i = 0; i < 9; i++) {
        double fi = f[idx_f(x, y, i)];
        r  += fi;
        mx += d_ex[i] * fi;
        my += d_ey[i] * fi;
    }
    rho[c] = r;
    if (r > 0.0) { ux[c] = mx / r; uy[c] = my / r; }
    else         { ux[c] = 0.0;     uy[c] = 0.0;     }
    if (obstacle[c]) { ux[c] = 0.0; uy[c] = 0.0; }
}

/* ------------------------------------------------------------------ *
 * Kernel 2: BGK collision — relax each cell's f toward equilibrium.  *
 * Purely local: reads rho/ux/uy computed in kernel 1, modifies f.    *
 * ------------------------------------------------------------------ */
__global__ void collision_kernel(double *f, const double *rho,
                                 const double *ux, const double *uy,
                                 const int *obstacle)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    int c = idx_c(x, y);
    if (obstacle[c]) return;

    double r = rho[c], vx = ux[c], vy = uy[c];
    for (int i = 0; i < 9; i++) {
        int fi_idx = idx_f(x, y, i);
        double feq = feq_i(i, r, vx, vy);
        f[fi_idx] -= (1.0 / TAU) * (f[fi_idx] - feq);
    }
}

/* ------------------------------------------------------------------ *
 * Kernel 3: streaming with bounce-back.                              *
 * Pull-style: thread at (x,y) reads from neighbour (x-ex, y-ey).    *
 * Each thread writes only to its own location in fnew — no conflicts.*
 * ------------------------------------------------------------------ */
__global__ void streaming_kernel(const double *f, double *fnew,
                                  const int *obstacle)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    int c = idx_c(x, y);
    if (obstacle[c]) {
        for (int i = 0; i < 9; i++) fnew[idx_f(x, y, i)] = 0.0;
        return;
    }

    for (int i = 0; i < 9; i++) {
        int xs = x - d_ex[i];
        int ys = y - d_ey[i];

        /* periodic wrap in x */
        if (xs < 0)    xs += NX;
        if (xs >= NX)  xs -= NX;

        if (ys < 0 || ys >= NY) {
            fnew[idx_f(x, y, i)] = f[idx_f(x, y, d_opp[i])];
            continue;
        }

        if (obstacle[idx_c(xs, ys)]) {
            fnew[idx_f(x, y, i)] = f[idx_f(x, y, d_opp[i])];
        } else {
            fnew[idx_f(x, y, i)] = f[idx_f(xs, ys, i)];
        }
    }
}

/* ------------------------------------------------------------------ *
 * Kernel 4: Zou-He velocity inlet at x = 0.                         *
 * 1D kernel — one thread per y row (y = 1..NY-2).                   *
 * ------------------------------------------------------------------ */
__global__ void inlet_kernel(double *fnew, const int *obstacle)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x + 1;  /* y from 1 to NY-2 */
    if (y >= NY - 1) return;

    int c = idx_c(0, y);
    if (obstacle[c]) return;

    double f0 = fnew[idx_f(0, y, 0)];
    double f2 = fnew[idx_f(0, y, 2)];
    double f4 = fnew[idx_f(0, y, 4)];
    double f3 = fnew[idx_f(0, y, 3)];
    double f6 = fnew[idx_f(0, y, 6)];
    double f7 = fnew[idx_f(0, y, 7)];

    double rho_in = (f0 + f2 + f4 + 2.0*(f3 + f6 + f7)) / (1.0 - U_INLET);

    fnew[idx_f(0, y, 1)] = f3 + (2.0/3.0) * rho_in * U_INLET;
    fnew[idx_f(0, y, 5)] = f7 + 0.5*(f4 - f2) + (1.0/6.0) * rho_in * U_INLET;
    fnew[idx_f(0, y, 8)] = f6 + 0.5*(f2 - f4) + (1.0/6.0) * rho_in * U_INLET;
}

/* ------------------------------------------------------------------ *
 * Kernel 5: zero-gradient outlet at x = NX-1.                       *
 * 1D kernel — copies from x=NX-2 to x=NX-1.                        *
 * ------------------------------------------------------------------ */
__global__ void outlet_kernel(double *fnew, const int *obstacle)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (y >= NY - 1) return;

    int co = idx_c(NX-1, y);
    if (obstacle[co]) return;

    for (int i = 0; i < 9; i++) {
        fnew[idx_f(NX-1, y, i)] = fnew[idx_f(NX-2, y, i)];
    }
}

/* ================================================================== */
/* HOST: PGM and PPM output (identical to OpenMP version)             */
/* ================================================================== */
static void save_pgm(int step,
                     const double *ux, const double *uy,
                     const int    *obstacle)
{
    char fname[80];
    snprintf(fname, sizeof(fname), FRAMES_DIR "/frame_%05d.pgm", step);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { perror(fname); return; }

    double umax = 1e-12;
    for (int x = 0; x < NX; x++)
        for (int y = 0; y < NY; y++) {
            int c = idx_c(x, y);
            double m = sqrt(ux[c]*ux[c] + uy[c]*uy[c]);
            if (m > umax) umax = m;
        }

    fprintf(fp, "P5\n%d %d\n255\n", NX, NY);
    for (int y = NY-1; y >= 0; y--)
        for (int x = 0; x < NX; x++) {
            int c = idx_c(x, y);
            unsigned char px;
            if (obstacle[c]) px = 0;
            else {
                double m = sqrt(ux[c]*ux[c] + uy[c]*uy[c]);
                int v = (int)(255.0 * m / umax);
                if (v < 0) v = 0; if (v > 255) v = 255;
                px = (unsigned char)v;
            }
            fputc(px, fp);
        }
    fclose(fp);
}

static void save_ppm_vorticity(int step,
                               const double *ux, const double *uy,
                               const int    *obstacle)
{
    char fname[80];
    snprintf(fname, sizeof(fname), FRAMES_PPM_DIR "/frame_%05d.ppm", step);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { perror(fname); return; }

    double *omega = (double *)malloc(NX * NY * sizeof(double));
    if (!omega) { fclose(fp); return; }

    double omax = 1e-12;
    for (int x = 0; x < NX; x++)
        for (int y = 0; y < NY; y++) {
            if (obstacle[idx_c(x, y)]) { omega[idx_c(x, y)] = 0.0; continue; }
            int xp = (x < NX-1) ? x+1 : x,  xm = (x > 0) ? x-1 : x;
            int yp = (y < NY-1) ? y+1 : y,  ym = (y > 0) ? y-1 : y;
            double w = (uy[idx_c(xp,y)] - uy[idx_c(xm,y)]) / (double)(xp - xm)
                     - (ux[idx_c(x,yp)] - ux[idx_c(x,ym)]) / (double)(yp - ym);
            omega[idx_c(x, y)] = w;
            if (fabs(w) > omax) omax = fabs(w);
        }

    fprintf(fp, "P6\n%d %d\n255\n", NX, NY);
    for (int y = NY-1; y >= 0; y--)
        for (int x = 0; x < NX; x++) {
            unsigned char r, g, b;
            if (obstacle[idx_c(x, y)]) { r = g = b = 30; }
            else {
                double t = omega[idx_c(x, y)] / omax;
                if (t >= 0.0) {
                    r = 255;
                    g = (unsigned char)(255.0 * (1.0 - t));
                    b = (unsigned char)(255.0 * (1.0 - t));
                } else {
                    r = (unsigned char)(255.0 * (1.0 + t));
                    g = (unsigned char)(255.0 * (1.0 + t));
                    b = 255;
                }
            }
            fputc(r, fp); fputc(g, fp); fputc(b, fp);
        }
    free(omega);
    fclose(fp);
}

/* ================================================================== */
/* Main                                                               */
/* ================================================================== */
int main(int argc, char **argv) {
    /* Create output directories */
    mkdir("frames", 0755);
    mkdir(FRAMES_DIR, 0755);
    mkdir(FRAMES_PPM_DIR, 0755);

    /* Parse optional block size from command line */
    int bx = DEFAULT_BX, by = DEFAULT_BY;
    if (argc >= 3) {
        bx = atoi(argv[1]);
        by = atoi(argv[2]);
        if (bx <= 0 || by <= 0 || bx * by > 1024) {
            fprintf(stderr, "Invalid block size %dx%d (max 1024 threads/block)\n",
                    bx, by);
            return 1;
        }
    }

    /* -------------------------------------------------------------- */
    /* Copy D2Q9 constants to GPU __constant__ memory                 */
    /* -------------------------------------------------------------- */
    CUDA_CHECK(cudaMemcpyToSymbol(d_ex,  h_ex,  9 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_ey,  h_ey,  9 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_w,   h_w,   9 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_opp, h_opp, 9 * sizeof(int)));

    /* -------------------------------------------------------------- */
    /* Host allocation and initialisation                             */
    /* -------------------------------------------------------------- */
    size_t Ncells = (size_t)NX * NY;
    size_t Nf     = Ncells * 9;

    double *h_f        = (double *)malloc(Nf     * sizeof(double));
    double *h_rho      = (double *)malloc(Ncells * sizeof(double));
    double *h_ux       = (double *)malloc(Ncells * sizeof(double));
    double *h_uy       = (double *)malloc(Ncells * sizeof(double));
    int    *h_obstacle = (int    *)malloc(Ncells * sizeof(int));

    for (int x = 0; x < NX; x++)
        for (int y = 0; y < NY; y++) {
            int c = idx_c(x, y);
            h_rho[c] = 1.0;
            double pert = 1.0 + 0.02 * cos(2.0 * M_PI * x / (double)NX * 4.0);
            h_ux[c]  = U_INLET * pert;
            h_uy[c]  = 0.0;
            int dx_ = x - CYL_X, dy_ = y - CYL_Y;
            h_obstacle[c] = (dx_*dx_ + dy_*dy_ <= CYL_R*CYL_R) ? 1 : 0;
            if (y == 0 || y == NY-1) h_obstacle[c] = 1;
            for (int i = 0; i < 9; i++)
                h_f[idx_f(x, y, i)] = feq_i_host(i, h_rho[c], h_ux[c], h_uy[c]);
        }

    /* -------------------------------------------------------------- */
    /* Device allocation (Step 2 from lecture workflow)                */
    /* -------------------------------------------------------------- */
    double *d_f, *d_fnew, *d_rho, *d_ux, *d_uy;
    int    *d_obstacle;

    CUDA_CHECK(cudaMalloc(&d_f,        Nf     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fnew,     Nf     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho,      Ncells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ux,       Ncells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_uy,       Ncells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_obstacle, Ncells * sizeof(int)));

    /* -------------------------------------------------------------- */
    /* Copy initial data host → device (Step 3 from lecture workflow)  */
    /* -------------------------------------------------------------- */
    CUDA_CHECK(cudaMemcpy(d_f,        h_f,        Nf     * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obstacle, h_obstacle, Ncells * sizeof(int),
                          cudaMemcpyHostToDevice));
    /* d_fnew is only written by the streaming kernel, no init needed */

    /* -------------------------------------------------------------- */
    /* Kernel launch configuration                                    */
    /* 2D grid: one thread per cell, block size bx × by.              */
    /* Grid dimensions are ceil(NX/bx) × ceil(NY/by) blocks.          */
    /* -------------------------------------------------------------- */
    dim3 block2D(bx, by);
    dim3 grid2D((NX + bx - 1) / bx, (NY + by - 1) / by);

    /* 1D grid for boundary kernels (NY-2 threads for y=1..NY-2) */
    int bc_threads = 256;
    int bc_blocks  = (NY - 2 + bc_threads - 1) / bc_threads;

    printf("D2Q9-BGK LBM CUDA: %dx%d lattice, %d steps, tau=%.2f, U=%.3f\n",
           NX, NY, NSTEPS, TAU, U_INLET);
    printf("Block size: %dx%d = %d threads/block\n", bx, by, bx * by);
    printf("Grid:       %dx%d blocks\n", grid2D.x, grid2D.y);
    printf("Reynolds number ~ %.1f\n",
           U_INLET * (2.0*CYL_R) / ((TAU - 0.5) / 3.0));
    fflush(stdout);

    /* -------------------------------------------------------------- */
    /* CUDA events for accurate GPU timing                            */
    /* -------------------------------------------------------------- */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    /* ============================================================== */
    /* Main time-stepping loop (Step 4: launch kernels on GPU)        */
    /* ============================================================== */
    for (int t = 1; t <= NSTEPS; t++) {

        /* Step 1: macroscopic variables — 2D grid, one thread per cell */
        macroscopic_kernel<<<grid2D, block2D>>>(d_f, d_rho, d_ux, d_uy,
                                                 d_obstacle);

        /* Step 2: BGK collision — same 2D grid */
        collision_kernel<<<grid2D, block2D>>>(d_f, d_rho, d_ux, d_uy,
                                               d_obstacle);

        /* Step 3: streaming with bounce-back — reads f, writes fnew */
        streaming_kernel<<<grid2D, block2D>>>(d_f, d_fnew, d_obstacle);

        /* Step 4: Zou-He inlet at x=0 — 1D grid over y */
        inlet_kernel<<<bc_blocks, bc_threads>>>(d_fnew, d_obstacle);

        /* Step 5: zero-gradient outlet at x=NX-1 — 1D grid over y */
        outlet_kernel<<<bc_blocks, bc_threads>>>(d_fnew, d_obstacle);

        /* Step 6: swap device pointers (no data movement!) */
        double *tmp = d_f; d_f = d_fnew; d_fnew = tmp;

        /* Step 7: periodic output — copy data back to host for PGM */
        if (t % OUTPUT_INTERVAL == 0) {
            /* Recompute macros on GPU before copying */
            macroscopic_kernel<<<grid2D, block2D>>>(d_f, d_rho, d_ux, d_uy,
                                                     d_obstacle);

            CUDA_CHECK(cudaMemcpy(h_ux, d_ux, Ncells * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_uy, d_uy, Ncells * sizeof(double),
                                  cudaMemcpyDeviceToHost));

            save_pgm(t, h_ux, h_uy, h_obstacle);
            save_ppm_vorticity(t, h_ux, h_uy, h_obstacle);
            printf("  step %5d / %d  saved frame\n", t, NSTEPS);
            fflush(stdout);
        }
    }

    /* -------------------------------------------------------------- */
    /* Stop timing and report results                                 */
    /* -------------------------------------------------------------- */
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    double elapsed = elapsed_ms / 1000.0;
    double mlups   = ((double)NSTEPS * NX * NY) / elapsed / 1.0e6;

    printf("\nDone.\n");
    printf("Block size: %dx%d\n", bx, by);
    printf("Elapsed:    %.3f s\n", elapsed);
    printf("Throughput: %.2f MLUPS\n", mlups);

    /* -------------------------------------------------------------- */
    /* Clean up (Step 6 from lecture workflow: free everything)        */
    /* -------------------------------------------------------------- */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    CUDA_CHECK(cudaFree(d_f));
    CUDA_CHECK(cudaFree(d_fnew));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_ux));
    CUDA_CHECK(cudaFree(d_uy));
    CUDA_CHECK(cudaFree(d_obstacle));

    free(h_f); free(h_rho); free(h_ux); free(h_uy); free(h_obstacle);

    return 0;
}
