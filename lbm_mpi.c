#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#ifdef _WIN32
#  include <direct.h>
#  define MKDIR(d) _mkdir(d)
#else
#  include <sys/stat.h>
#  define MKDIR(d) mkdir((d), 0755)
#endif

#define FRAMES_DIR     "frames/mpi"
#define FRAMES_PPM_DIR "frames/mpi_ppm"

#define NX               600
#define NY               200
#define NSTEPS           10000
#define OUTPUT_INTERVAL  100

#define U_INLET          0.1
#define TAU              0.6

#define CYL_X            (NX/4)
#define CYL_Y            (NY/2)
#define CYL_R            15

/* D2Q9 velocities, weights, and opposite-direction indices */
static const int    ex[9]  = {  0,  1,  0, -1,  0,  1, -1, -1,  1 };
static const int    ey[9]  = {  0,  0,  1,  0, -1,  1,  1, -1, -1 };
static const double w_[9]  = { 4.0/9.0,
                               1.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0,
                               1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
static const int    opp[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

/* local width (set at runtime): local_nx + 2 ghost columns */
static int L_NX;

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

static void save_pgm(int step,
                     const double *ux_full, const double *uy_full,
                     const int    *obs_full)
{
    char fname[80];
    snprintf(fname, sizeof(fname), FRAMES_DIR "/frame_%05d.pgm", step);
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

/* vorticity: omega = duy/dx - dux/dy, mapped red/white/blue */
static void save_ppm_vorticity(int step,
                               const double *ux_full, const double *uy_full,
                               const int    *obs_full)
{
    char fname[80];
    snprintf(fname, sizeof(fname), FRAMES_PPM_DIR "/frame_%05d.ppm", step);
    FILE *fp = fopen(fname, "wb");
    if (!fp) { perror(fname); return; }

    double *omega = (double *)malloc(NX * NY * sizeof(double));
    if (!omega) { fclose(fp); return; }

    double omax = 1e-12;
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            int c = x * NY + y;
            if (obs_full[c]) { omega[c] = 0.0; continue; }
            int xp = (x < NX-1) ? x+1 : x,  xm = (x > 0) ? x-1 : x;
            int yp = (y < NY-1) ? y+1 : y,  ym = (y > 0) ? y-1 : y;
            double w = (uy_full[xp*NY+y] - uy_full[xm*NY+y]) / (double)(xp - xm)
                     - (ux_full[x*NY+yp] - ux_full[x*NY+ym]) / (double)(yp - ym);
            omega[c] = w;
            if (fabs(w) > omax) omax = fabs(w);
        }
    }

    fprintf(fp, "P6\n%d %d\n255\n", NX, NY);
    for (int y = NY-1; y >= 0; y--) {
        for (int x = 0; x < NX; x++) {
            int c = x * NY + y;
            unsigned char r, g, b;
            if (obs_full[c]) {
                r = g = b = 30;
            } else {
                double t = omega[c] / omax;
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
    }
    free(omega);
    fclose(fp);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        MKDIR("frames");
        MKDIR(FRAMES_DIR);
        MKDIR(FRAMES_PPM_DIR);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* 1D domain decomposition along x */
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

    L_NX = local_nx + 2;   /* real columns + 2 ghost columns */

    int left_rank  = (rank > 0)           ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (rank < nprocs - 1)  ? rank + 1 : MPI_PROC_NULL;

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

    /* exchange obstacle ghost columns once (static, never changes) */
    MPI_Sendrecv(&obstacle[idx_c(local_nx, 0)], NY, MPI_INT, right_rank, 0,
                 &obstacle[idx_c(0, 0)],        NY, MPI_INT, left_rank,  0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&obstacle[idx_c(1, 0)],        NY, MPI_INT, left_rank,  1,
                 &obstacle[idx_c(local_nx+1, 0)],NY, MPI_INT, right_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* rank 0 output buffers */
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

    int my_count = local_nx * NY;
    MPI_Gather(&my_count, 1, MPI_INT,
               recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < nprocs; r++)
            displs[r] = displs[r-1] + recvcounts[r-1];
    }

    /* gather obstacle map on rank 0 */
    {
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

    int halo_size = NY * 9;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int t = 1; t <= NSTEPS; t++) {

        /* 1. macroscopic variables */
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

        /* 2. BGK collision */
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

        /* halo exchange: send post-collision f to neighbours for streaming */
        MPI_Sendrecv(&f[idx_f(local_nx, 0, 0)], halo_size, MPI_DOUBLE,
                     right_rank, 10,
                     &f[idx_f(0, 0, 0)],         halo_size, MPI_DOUBLE,
                     left_rank,  10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&f[idx_f(1, 0, 0)],          halo_size, MPI_DOUBLE,
                     left_rank,  11,
                     &f[idx_f(local_nx+1, 0, 0)], halo_size, MPI_DOUBLE,
                     right_rank, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* 3. streaming + bounce-back */
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

                    if (lxs < 0)             lxs = 0;
                    if (lxs > local_nx + 1)  lxs = local_nx + 1;

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

        /* 4. Zou-He inlet at global x = 0 (rank 0 only, local lx = 1) */
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

        /* 5. zero-gradient outlet at global x = NX-1 (last rank only) */
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

        /* 6. swap buffers */
        double *tmp = f; f = fnew; fnew = tmp;

        /* 7. output: gather to rank 0, write frames */
        if (t % OUTPUT_INTERVAL == 0) {
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
                save_ppm_vorticity(t, ux_full, uy_full, obs_full);
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

    free(f); free(fnew);
    free(rho); free(ux); free(uy); free(obstacle);
    if (rank == 0) {
        free(ux_full); free(uy_full); free(obs_full);
        free(recvcounts); free(displs);
    }

    MPI_Finalize();
    return 0;
}
