// data common between needed by  kernels_c.c


// Global grid sizes, start-end indices for local grids,  MPI topology
extern int  ngxyz[3], nbxyz[3], sx, ex, sy, ey, sz, ez, nlx, nly, nlz, npxyz[3];

// OpenMP threads number
extern int nthreads;

// solution array
extern Real  *udata, *uOld, *uNew;

inline int uindex(const struct grid_info_t *grid, const int i, const int j, const int k);
void stencil_update(const struct grid_info_t *grid, int s1, int e1, int s2, int e2, int s3, int e3, double * x); // needed by MPI 


