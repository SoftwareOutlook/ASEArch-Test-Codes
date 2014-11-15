// data common between needed by  kernels_c.c
/* utils exported valriables

 Lucian Anton
 July 2013
*/

// Number of iteration 
extern int nruns, niter;

// OpenMP threads number
extern int nthreads;

// solution array
extern Real *udata, *uOld, *uNew;

// IO control
extern int pContext, testComputation;

void initContext(int argc, char *argv[], struct grid_info_t * grid, int *kernel_key);
void setPEsParams(struct grid_info_t *g);
void initialise_grid( const struct grid_info_t *g);
void printContext(const struct grid_info_t *g);
void check_norm(const struct grid_info_t *g, int iter, double norm);
void statistics(const struct grid_info_t *g, const struct times_t *times,  
                struct times_t *minTime, struct times_t *meanTime, struct times_t *maxTime);

void stdoutIO( const struct grid_info_t *g, const struct times_t *times,  
              const struct times_t *minTime, const struct times_t *meanTime, const struct times_t *maxTime, 
	       double norm);
double local_norm(const struct grid_info_t *g);
void timeUpdate(struct times_t *times);
double my_wtime();
void error_abort(const char *s1, const char *s2);


/**
 * InitialiseGPUData
 * this function handles looking for GPU in the system and if it exists
 * it initialises required host and device memory for computations
 * and further initialises them with data from grid arrays and finally copying GPU array to perform
 * GPU computation
 */
void initialiseGPUData(int NX,int NY,int NZ);
/**
 * free GPU memory
 */
void freeDeviceMemory();





