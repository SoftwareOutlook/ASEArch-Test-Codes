// data common between needed by  kernels_c.c
/* utils exported valriables

 Lucian Anton
 July 2013
*/

// Number of iteration 
extern int niter;

// OpenMP threads number
extern int nthreads;

// solution array
extern Real *udata, *uOld, *uNew;

// IO control
extern int pContext, testComputation;

void initContext(int argc, char *argv[], struct grid_info_t * grid, int *kernel_key);
void setPEsParams(struct grid_info_t *g);
void initialise_grid( const struct grid_info_t *g);
void printContext(const struct grid_info_t *g, int kernel_key);
void check_norm(const struct grid_info_t *g, int iter, double norm);
void statistics(const struct grid_info_t *g, double *times,  
                double *minTime, double *meanTime, double *maxTime,
                double *stdvTime, double *NstdvTime);
void stdoutIO( const struct grid_info_t *g, const int kernel_key, const double *times,  
              double minTime, double meanTime, double maxTime, 
	       double NstdvTime, double norm);
double local_norm(const struct grid_info_t *g);
void timeUpdate(double *times);
double my_wtime();




