/*
 * lbm_mpi.c
 *
 * MPI-parallelised D2Q9-BGK Lattice Boltzmann Method:
 * 2D incompressible flow past a circular cylinder.
 *
 * Parallelisation strategy:
 *   - 1D domain decomposition along the x-axis.  The 600-column
 *     lattice is split into contiguous strips, one per MPI process.
 *   - Each process allocates two extra "ghost" columns (one left,
 *     one right) to hold neighbour data needed by the streaming step.
 *   - After the collision step, every process exchanges its outermost
 *     real column of f-values with its neighbour via MPI_Sendrecv.
 *     This is a single call that sends and receives simultaneously,
 *     avoiding deadlock (a key concern from the lectures).
 *   - Boundary conditions:  rank 0 owns x=0 (Zou-He inlet);
 *     the last rank owns x=NX-1 (zero-gradient outlet).
 *   - For PGM output, local velocity arrays are gathered to rank 0
 *     with MPI_Gatherv (variable counts for unequal splits).
 *
 * Build:   mpicc -O3 -o lbm_mpi lbm_mpi.c -lm
 * Run:     mpirun -np 4 ./lbm_mpi
 *
 * References (algorithmic only; this code is original):
 *   - Mocz (2020), Palabos LBM codes, Krüger et al. (2017).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

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

/* ------------------------------------------------------------------ */
/* D2Q9 lattice constants                                             */
/* ------------------------------------------------------------------ */
static const int    ex[9]  = {  0,  1,  0, -1,  0,  1, -1, -1,  1 };
static const int    ey[9]  = {  0,  0,  1,  0, -1,  1,  1, -1, -1 };
static const double w_[9]  = { 4.0/9.0,
                               1.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0,
                               1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
static const int    opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

/* ------------------------------------------------------------------ */
/* Index helpers — local grid has (local_nx + 2) columns (with ghosts)*/
/* lx = 0 is left ghost, lx = local_nx+1 is right ghost.             */
/* Real data sits in lx = 1 .. local_nx.                              */
/* ------------------------------------------------------------------ */
static int L_NX;   /* set at runtime = local_nx + 2 (total local width) */

static inline int idx_f(int lx, int y, int i) {
    return ((lx * NY) + y) * 9 + i;
}
static inline int idx_c(int lx, int y) {
    return lx * NY + y;
}

static inline double feq_i(int i, double rho, double ux, double uy) {
    double cu  = 3.0 * (ex[i]*ux + ey[i]*uy);
    double usq = 1.5 * (ux*ux + uy*uy);
    return w_[i] * rho * (1.0 + cu + 0.5*cu*cu - usq);
}

/* ------------------------------------------------------------------ */
/* Save PGM from the full-domain arrays on rank 0                     */
/* ------------------------------------------------------------------ */
static void save_pgm(int step,
                     const double *ux_full, const double *uy_full,
                     const int    *obs_full)
{
    char fname[64];
    snprintf(fname, sizeof(fname), "frame_%05d.pgm", step);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { perror(fname); return; }

    double umax = 1e-12;
    for (int x = 0; x < NX; x++)
        for (int y = 0; y < NY; y++) {
            int c = x * NY + y;
            double m = sqrt(ux_full[c]*ux_full[c] + uy_full[c]*uy_full[c]);
            if (m > umax) umax = m;
        }

    fprintf(fp, "P5\n%d %d\n255\n", NX, NY);
    for (int y = NY-1; y >= 0; y--)
        for (int x = 0; x < NX; x++) {
            int c = x * NY + y;
            unsigned char px;
            if (obs_full[c]) { px = 0; }
            else {
                double m = sqrt(ux_full[c]*ux_full[c] + uy_full[c]*uy_full[c]);
                int v = (int)(255.0 * m / umax);
                if (v < 0) v = 0; if (v > 255) v = 255;
                px = (unsigned char)v;
            }
            fputc(px, fp);
        }
    fclose(fp);
}

/* ================================================================== */
/* Main                                                               */
/* ================================================================== */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* -------------------------------------------------------------- */
    /* 1D domain decomposition along x.                               */
    /* Each rank gets a contiguous strip of columns.                   */
    /* base_nx = NX / nprocs columns each; the last rank gets the     */
    /* remainder so NX need not be evenly divisible.                   */
    /* -------------------------------------------------------------- */
    int base_nx  = NX / nprocs;
    int leftover = NX % nprocs;
    int local_nx, x_start;

    if (rank < leftover) {
        local_nx = base_nx + 1;
        x_start  = rank * (base_nx + 1);
    } else {
        local_nx = base_nx;
        x_start  = leftover * (base_nx + 1) + (rank - leftover) * base_nx;
    }

    L_NX = local_nx + 2;   /* total local width including 2 ghost columns */

    /* Identify left and right neighbours (MPI_PROC_NULL = no neighbour) */
    int left_rank  = (rank > 0)           ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (rank < nprocs - 1)  ? rank + 1 : MPI_PROC_NULL;

    /* -------------------------------------------------------------- */
    /* Allocate local arrays.                                         */
    /* Dimensions: L_NX columns × NY rows (including ghost columns).  */
    /* -------------------------------------------------------------- */
    size_t Ncells_local = (size_t)L_NX * NY;
    size_t Nf_local     = Ncells_local * 9;

    double *f        = (double *)malloc(Nf_local     * sizeof(double));
    double *fnew     = (double *)malloc(Nf_local     * sizeof(double));
    double *rho      = (double *)malloc(Ncells_local * sizeof(double));
    double *ux       = (double *)malloc(Ncells_local * sizeof(double));
    double *uy       = (double *)malloc(Ncells_local * sizeof(double));
    int    *obstacle = (int    *)malloc(Ncells_local * sizeof(int));

    if (!f || !fnew || !rho || !ux || !uy || !obstacle) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* -------------------------------------------------------------- */
    /* Initialise local portion.                                      */
    /* lx = 1..local_nx are the real cells; ghosts are set to zero.   */
    /* -------------------------------------------------------------- */
    memset(f,   0, Nf_local     * sizeof(double));
    memset(fnew,0, Nf_local     * sizeof(double));
    memset(obstacle, 0, Ncells_local * sizeof(int));

    for (int lx = 1; lx <= local_nx; lx++) {
        int gx = x_start + (lx - 1);   /* global x coordinate */
        for (int y = 0; y < NY; y++) {
            int c = idx_c(lx, y);
            rho[c] = 1.0;
            double pert = 1.0 + 0.02 * cos(2.0 * M_PI * gx / (double)NX * 4.0);
            ux[c]  = U_INLET * pert;
            uy[c]  = 0.0;

            int dx_ = gx - CYL_X;
            int dy_ = y  - CYL_Y;
            obstacle[c] = (dx_*dx_ + dy_*dy_ <= CYL_R*CYL_R) ? 1 : 0;
            if (y == 0 || y == NY-1) obstacle[c] = 1;

            for (int i = 0; i < 9; i++)
                f[idx_f(lx, y, i)] = feq_i(i, rho[c], ux[c], uy[c]);
        }
    }

    /* We also need the obstacle array in ghost columns for streaming  */
    /* Exchange obstacle data once (it never changes).                 */
    MPI_Sendrecv(&obstacle[idx_c(local_nx, 0)], NY, MPI_INT, right_rank, 0,
                 &obstacle[idx_c(0, 0)],        NY, MPI_INT, left_rank,  0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&obstacle[idx_c(1, 0)],        NY, MPI_INT, left_rank,  1,
                 &obstacle[idx_c(local_nx+1, 0)],NY, MPI_INT, right_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* -------------------------------------------------------------- */
    /* Prepare output gathering arrays (rank 0 only).                 */
    /* -------------------------------------------------------------- */
    double *ux_full = NULL, *uy_full = NULL;
    int    *obs_full = NULL;
    int    *recvcounts = NULL, *displs = NULL;

    if (rank == 0) {
        ux_full  = (double *)malloc((size_t)NX * NY * sizeof(double));
        uy_full  = (double *)malloc((size_t)NX * NY * sizeof(double));
        obs_full = (int    *)malloc((size_t)NX * NY * sizeof(int));
        recvcounts = (int *)malloc(nprocs * sizeof(int));
        displs     = (int *)malloc(nprocs * sizeof(int));
    }

    /* Every rank computes its own count = local_nx * NY */
    int my_count = local_nx * NY;
    /* Gather all counts to rank 0 for MPI_Gatherv */
    MPI_Gather(&my_count, 1, MPI_INT,
               recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < nprocs; r++)
            displs[r] = displs[r-1] + recvcounts[r-1];
    }

    /* Build full obstacle map on rank 0 (once — it doesn't change) */
    {
        /* Pack local obstacle data (real cells only, contiguous) into a buffer */
        int *obs_local = (int *)malloc(my_count * sizeof(int));
        for (int lx = 1; lx <= local_nx; lx++)
            for (int y = 0; y < NY; y++)
                obs_local[(lx-1)*NY + y] = obstacle[idx_c(lx, y)];

        MPI_Gatherv(obs_local, my_count, MPI_INT,
                    obs_full, recvcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);
        free(obs_local);
    }

    if (rank == 0) {
        printf("D2Q9-BGK LBM MPI: %dx%d lattice, %d steps, tau=%.2f, U=%.3f\n",
               NX, NY, NSTEPS, TAU, U_INLET);
        printf("Processes: %d\n", nprocs);
        printf("Reynolds number ~ %.1f\n",
               U_INLET * (2.0*CYL_R) / ((TAU - 0.5) / 3.0));
        fflush(stdout);
    }

    /* Allocate packing buffers for halo exchange.                     *
     * Each ghost column holds NY cells × 9 doubles = NY*9 doubles.   */
    int halo_size = NY * 9;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* ============================================================== */
    /* Main time-stepping loop                                        */
    /* ============================================================== */
    for (int t = 1; t <= NSTEPS; t++) {

        /* ---------------------------------------------------------- *
         * Step 1: macroscopic variables (local only, no communication)*
         * ---------------------------------------------------------- */
        for (int lx = 1; lx <= local_nx; lx++) {
            for (int y = 0; y < NY; y++) {
                int c = idx_c(lx, y);
                double r = 0.0, mx = 0.0, my = 0.0;
                for (int i = 0; i < 9; i++) {
                    double fi = f[idx_f(lx, y, i)];
                    r += fi; mx += ex[i]*fi; my += ey[i]*fi;
                }
                rho[c] = r;
                if (r > 0.0) { ux[c] = mx/r; uy[c] = my/r; }
                else         { ux[c] = 0.0;   uy[c] = 0.0;   }
                if (obstacle[c]) { ux[c] = 0.0; uy[c] = 0.0; }
            }
        }

        /* ---------------------------------------------------------- *
         * Step 2: collision (local only — each cell updates its own f)*
         * ---------------------------------------------------------- */
        for (int lx = 1; lx <= local_nx; lx++) {
            for (int y = 0; y < NY; y++) {
                int c = idx_c(lx, y);
                if (obstacle[c]) continue;
                double r = rho[c], vx = ux[c], vy = uy[c];
                for (int i = 0; i < 9; i++) {
                    int fi_idx = idx_f(lx, y, i);
                    double feq = feq_i(i, r, vx, vy);
                    f[fi_idx] -= (1.0/TAU) * (f[fi_idx] - feq);
                }
            }
        }

        /* ---------------------------------------------------------- *
         * Halo exchange: send post-collision f values to neighbours.  *
         *                                                             *
         * After collision, the streaming step needs to pull f from    *
         * neighbours.  A cell at local column 1 may pull from column  *
         * 0 (the left ghost), which must contain the right-boundary   *
         * data from the left-neighbour process.  Likewise, a cell at  *
         * local_nx may pull from local_nx+1 (the right ghost).       *
         *                                                             *
         * MPI_Sendrecv sends and receives in one call, avoiding the   *
         * deadlock that would occur if everyone tried MPI_Send first. *
         * ---------------------------------------------------------- */

        /* Send my rightmost real column → right neighbour's left ghost *
         * Receive from left neighbour's rightmost real → my left ghost */
        MPI_Sendrecv(&f[idx_f(local_nx, 0, 0)], halo_size, MPI_DOUBLE,
                     right_rank, 10,
                     &f[idx_f(0, 0, 0)],         halo_size, MPI_DOUBLE,
                     left_rank,  10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Send my leftmost real column → left neighbour's right ghost *
         * Receive from right neighbour's leftmost real → my right ghost */
        MPI_Sendrecv(&f[idx_f(1, 0, 0)],          halo_size, MPI_DOUBLE,
                     left_rank,  11,
                     &f[idx_f(local_nx+1, 0, 0)], halo_size, MPI_DOUBLE,
                     right_rank, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* ---------------------------------------------------------- *
         * Step 3: streaming with bounce-back (local, uses ghosts)    *
         * ---------------------------------------------------------- */
        for (int lx = 1; lx <= local_nx; lx++) {
            for (int y = 0; y < NY; y++) {
                int c = idx_c(lx, y);
                if (obstacle[c]) {
                    for (int i = 0; i < 9; i++) fnew[idx_f(lx, y, i)] = 0.0;
                    continue;
                }
                for (int i = 0; i < 9; i++) {
                    int lxs = lx - ex[i];
                    int ys  = y  - ey[i];

                    /* x-direction wrapping for global periodicity.     *
                     * Left boundary of rank 0: the left ghost was not  *
                     * filled (left_rank = MPI_PROC_NULL), so we wrap   *
                     * by treating the ghost as having outlet-copied     *
                     * data.  In practice the inlet BC overwrites x=0,  *
                     * so this just needs to not crash.                 */
                    if (lxs < 0)     lxs = 0;
                    if (lxs > local_nx + 1) lxs = local_nx + 1;

                    if (ys < 0 || ys >= NY) {
                        fnew[idx_f(lx, y, i)] = f[idx_f(lx, y, opp[i])];
                        continue;
                    }

                    if (obstacle[idx_c(lxs, ys)]) {
                        fnew[idx_f(lx, y, i)] = f[idx_f(lx, y, opp[i])];
                    } else {
                        fnew[idx_f(lx, y, i)] = f[idx_f(lxs, ys, i)];
                    }
                }
            }
        }

        /* ---------------------------------------------------------- *
         * Step 4: Zou-He inlet at global x = 0 (only rank 0).       *
         * In rank 0's local coordinates, global x=0 is lx=1.        *
         * ---------------------------------------------------------- */
        if (rank == 0) {
            for (int y = 1; y < NY-1; y++) {
                int c = idx_c(1, y);
                if (obstacle[c]) continue;

                double f0 = fnew[idx_f(1, y, 0)];
                double f2 = fnew[idx_f(1, y, 2)];
                double f4 = fnew[idx_f(1, y, 4)];
                double f3 = fnew[idx_f(1, y, 3)];
                double f6 = fnew[idx_f(1, y, 6)];
                double f7 = fnew[idx_f(1, y, 7)];

                double rho_in = (f0 + f2 + f4 + 2.0*(f3 + f6 + f7))
                              / (1.0 - U_INLET);

                fnew[idx_f(1, y, 1)] = f3 + (2.0/3.0)*rho_in*U_INLET;
                fnew[idx_f(1, y, 5)] = f7 + 0.5*(f4 - f2)
                                     + (1.0/6.0)*rho_in*U_INLET;
                fnew[idx_f(1, y, 8)] = f6 + 0.5*(f2 - f4)
                                     + (1.0/6.0)*rho_in*U_INLET;
            }
        }

        /* ---------------------------------------------------------- *
         * Step 5: zero-gradient outlet at global x = NX-1.           *
         * Last rank owns this; in its local coords, NX-1 is          *
         * lx = local_nx and NX-2 is lx = local_nx - 1.              *
         * ---------------------------------------------------------- */
        if (rank == nprocs - 1) {
            for (int y = 1; y < NY-1; y++) {
                int co = idx_c(local_nx, y);
                if (obstacle[co]) continue;
                for (int i = 0; i < 9; i++) {
                    fnew[idx_f(local_nx, y, i)] =
                        fnew[idx_f(local_nx - 1, y, i)];
                }
            }
        }

        /* Step 6: swap buffers */
        double *tmp = f; f = fnew; fnew = tmp;

        /* ---------------------------------------------------------- *
         * Step 7: periodic output — gather to rank 0, write PGM.    *
         * ---------------------------------------------------------- */
        if (t % OUTPUT_INTERVAL == 0) {
            /* Recompute macros for output */
            for (int lx = 1; lx <= local_nx; lx++) {
                for (int y = 0; y < NY; y++) {
                    int c = idx_c(lx, y);
                    if (obstacle[c]) { ux[c]=0; uy[c]=0; continue; }
                    double r=0, mx=0, my=0;
                    for (int i = 0; i < 9; i++) {
                        double fi = f[idx_f(lx, y, i)];
                        r += fi; mx += ex[i]*fi; my += ey[i]*fi;
                    }
                    if (r > 0) { ux[c] = mx/r; uy[c] = my/r; }
                    else       { ux[c] = 0;     uy[c] = 0;     }
                }
            }

            /* Pack local real cells into contiguous send buffers */
            double *ux_send = (double *)malloc(my_count * sizeof(double));
            double *uy_send = (double *)malloc(my_count * sizeof(double));
            for (int lx = 1; lx <= local_nx; lx++)
                for (int y = 0; y < NY; y++) {
                    int dst = (lx-1)*NY + y;
                    ux_send[dst] = ux[idx_c(lx, y)];
                    uy_send[dst] = uy[idx_c(lx, y)];
                }

            MPI_Gatherv(ux_send, my_count, MPI_DOUBLE,
                        ux_full, recvcounts, displs, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            MPI_Gatherv(uy_send, my_count, MPI_DOUBLE,
                        uy_full, recvcounts, displs, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            free(ux_send);
            free(uy_send);

            if (rank == 0) {
                save_pgm(t, ux_full, uy_full, obs_full);
                printf("  step %5d / %d  saved frame\n", t, NSTEPS);
                fflush(stdout);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;
    double mlups = ((double)NSTEPS * NX * NY) / elapsed / 1.0e6;

    if (rank == 0) {
        printf("\nDone.\n");
        printf("Processes:  %d\n", nprocs);
        printf("Elapsed:    %.3f s\n", elapsed);
        printf("Throughput: %.2f MLUPS\n", mlups);
    }

    /* Clean up */
    free(f); free(fnew);
    free(rho); free(ux); free(uy); free(obstacle);
    if (rank == 0) {
        free(ux_full); free(uy_full); free(obs_full);
        free(recvcounts); free(displs);
    }

    MPI_Finalize();
    return 0;
}