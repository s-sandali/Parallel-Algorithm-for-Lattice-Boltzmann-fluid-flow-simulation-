
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _WIN32
#  include <direct.h>
#  define MKDIR(d) _mkdir(d)
#else
#  include <sys/stat.h>
#  define MKDIR(d) mkdir((d), 0755)
#endif

#define FRAMES_DIR     "frames/serial"
#define FRAMES_PPM_DIR "frames/serial_ppm"

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

static inline int idx_f(int x, int y, int i) {
    return ((x * NY) + y) * 9 + i;
}
static inline int idx_c(int x, int y) {
    return x * NY + y;
}

static inline double feq_i(int i, double rho, double ux, double uy) {
    double cu  = 3.0 * (ex[i]*ux + ey[i]*uy);
    double usq = 1.5 * (ux*ux + uy*uy);
    return w_[i] * rho * (1.0 + cu + 0.5*cu*cu - usq);
}

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
    for (int y = NY-1; y >= 0; y--) {
        for (int x = 0; x < NX; x++) {
            int c = idx_c(x, y);
            unsigned char px;
            if (obstacle[c]) {
                px = 0;
            } else {
                double m = sqrt(ux[c]*ux[c] + uy[c]*uy[c]);
                int v = (int)(255.0 * m / umax);
                if (v < 0)   v = 0;
                if (v > 255) v = 255;
                px = (unsigned char)v;
            }
            fputc(px, fp);
        }
    }
    fclose(fp);
}

/* vorticity: omega = duy/dx - dux/dy, mapped red/white/blue */
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
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            if (obstacle[idx_c(x, y)]) { omega[idx_c(x, y)] = 0.0; continue; }
            int xp = (x < NX-1) ? x+1 : x,  xm = (x > 0) ? x-1 : x;
            int yp = (y < NY-1) ? y+1 : y,  ym = (y > 0) ? y-1 : y;
            double w = (uy[idx_c(xp,y)] - uy[idx_c(xm,y)]) / (double)(xp - xm)
                     - (ux[idx_c(x,yp)] - ux[idx_c(x,ym)]) / (double)(yp - ym);
            omega[idx_c(x, y)] = w;
            if (fabs(w) > omax) omax = fabs(w);
        }
    }

    fprintf(fp, "P6\n%d %d\n255\n", NX, NY);
    for (int y = NY-1; y >= 0; y--) {
        for (int x = 0; x < NX; x++) {
            unsigned char r, g, b;
            if (obstacle[idx_c(x, y)]) {
                r = g = b = 30;
            } else {
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
    }
    free(omega);
    fclose(fp);
}

int main(void) {
    MKDIR("frames");
    MKDIR(FRAMES_DIR);
    MKDIR(FRAMES_PPM_DIR);

    size_t Ncells = (size_t)NX * NY;
    size_t Nf     = Ncells * 9;

    double *f        = (double *)malloc(Nf     * sizeof(double));
    double *fnew     = (double *)malloc(Nf     * sizeof(double));
    double *rho      = (double *)malloc(Ncells * sizeof(double));
    double *ux       = (double *)malloc(Ncells * sizeof(double));
    double *uy       = (double *)malloc(Ncells * sizeof(double));
    int    *obstacle = (int    *)malloc(Ncells * sizeof(int));

    if (!f || !fnew || !rho || !ux || !uy || !obstacle) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            int c = idx_c(x, y);
            rho[c] = 1.0;
            double pert = 1.0 + 0.02 * cos(2.0 * M_PI * x / (double)NX * 4.0);
            ux[c]  = U_INLET * pert;
            uy[c]  = 0.0;

            int dx_ = x - CYL_X;
            int dy_ = y - CYL_Y;
            obstacle[c] = (dx_*dx_ + dy_*dy_ <= CYL_R*CYL_R) ? 1 : 0;
            if (y == 0 || y == NY-1) obstacle[c] = 1;

            for (int i = 0; i < 9; i++)
                f[idx_f(x, y, i)] = feq_i(i, rho[c], ux[c], uy[c]);
        }
    }

    printf("D2Q9-BGK LBM serial: %dx%d lattice, %d steps, tau=%.2f, U=%.3f\n",
           NX, NY, NSTEPS, TAU, U_INLET);
    printf("Reynolds number ~ %.1f\n",
           U_INLET * (2.0*CYL_R) / ((TAU - 0.5) / 3.0));
    fflush(stdout);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int t = 1; t <= NSTEPS; t++) {

        /* 1. macroscopic variables */
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                int c = idx_c(x, y);
                double r = 0.0, mx = 0.0, my = 0.0;
                for (int i = 0; i < 9; i++) {
                    double fi = f[idx_f(x, y, i)];
                    r  += fi;
                    mx += ex[i] * fi;
                    my += ey[i] * fi;
                }
                rho[c] = r;
                if (r > 0.0) { ux[c] = mx / r; uy[c] = my / r; }
                else         { ux[c] = 0.0;     uy[c] = 0.0;     }
                if (obstacle[c]) { ux[c] = 0.0; uy[c] = 0.0; }
            }
        }

        /* 2. BGK collision */
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                int c = idx_c(x, y);
                if (obstacle[c]) continue;
                double r  = rho[c], vx = ux[c], vy = uy[c];
                for (int i = 0; i < 9; i++) {
                    int    fi_idx = idx_f(x, y, i);
                    double feq    = feq_i(i, r, vx, vy);
                    f[fi_idx] -= (1.0/TAU) * (f[fi_idx] - feq);
                }
            }
        }

        /* 3. streaming + bounce-back (pull-style) */
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                int c = idx_c(x, y);
                if (obstacle[c]) {
                    for (int i = 0; i < 9; i++) fnew[idx_f(x, y, i)] = 0.0;
                    continue;
                }
                for (int i = 0; i < 9; i++) {
                    int xs = x - ex[i];
                    int ys = y - ey[i];

                    if (xs < 0)    xs += NX;
                    if (xs >= NX)  xs -= NX;

                    if (ys < 0 || ys >= NY) {
                        fnew[idx_f(x, y, i)] = f[idx_f(x, y, opp[i])];
                        continue;
                    }

                    if (obstacle[idx_c(xs, ys)]) {
                        fnew[idx_f(x, y, i)] = f[idx_f(x, y, opp[i])];
                    } else {
                        fnew[idx_f(x, y, i)] = f[idx_f(xs, ys, i)];
                    }
                }
            }
        }

        /* 4. Zou-He inlet at x = 0 */
        for (int y = 1; y < NY-1; y++) {
            int c = idx_c(0, y);
            if (obstacle[c]) continue;

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

        /* 5. zero-gradient outlet at x = NX-1 */
        for (int y = 1; y < NY-1; y++) {
            int co = idx_c(NX-1, y);
            if (obstacle[co]) continue;
            for (int i = 0; i < 9; i++)
                fnew[idx_f(NX-1, y, i)] = fnew[idx_f(NX-2, y, i)];
        }

        /* 6. swap buffers */
        double *tmp = f; f = fnew; fnew = tmp;

        /* 7. output */
        if (t % OUTPUT_INTERVAL == 0) {
            for (int x = 0; x < NX; x++) {
                for (int y = 0; y < NY; y++) {
                    int c = idx_c(x, y);
                    if (obstacle[c]) { ux[c] = 0.0; uy[c] = 0.0; continue; }
                    double r = 0.0, mx = 0.0, my = 0.0;
                    for (int i = 0; i < 9; i++) {
                        double fi = f[idx_f(x, y, i)];
                        r  += fi;
                        mx += ex[i] * fi;
                        my += ey[i] * fi;
                    }
                    if (r > 0.0) { ux[c] = mx/r; uy[c] = my/r; }
                    else         { ux[c] = 0.0;   uy[c] = 0.0;   }
                }
            }
            save_pgm(t, ux, uy, obstacle);
            save_ppm_vorticity(t, ux, uy, obstacle);
            printf("  step %5d / %d  saved frame\n", t, NSTEPS);
            fflush(stdout);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double mlups   = ((double)NSTEPS * NX * NY) / elapsed / 1.0e6;

    printf("\nDone.\n");
    printf("Elapsed:    %.3f s\n", elapsed);
    printf("Throughput: %.2f MLUPS\n", mlups);

    free(f); free(fnew);
    free(rho); free(ux); free(uy); free(obstacle);
    return 0;
}
