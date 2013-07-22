/* kernels headers
 
Lucian Anton
July 2013

*/

inline int uindex(const struct grid_info_t *grid, const int i, const int j, const int k);
void stencil_update(const struct grid_info_t *grid, int s1, int e1, int s2, int e2, int s3, int e3);
void laplace3d(const struct grid_info_t *g, int kernel_key, double *tstart, double *tend);
