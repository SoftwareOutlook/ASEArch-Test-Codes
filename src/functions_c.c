/*
This is the partof  DL_HOMB, a hybrid OpemMP MPI Benchmark for
3D Jacobi solver for Laplace equation.

Lucian Anton July 2013.

This code started from v 1.0 of HOMB
http://sourceforge.net/projects/homb/

Below is the original copyright and licence.



  Copyright 2009 Maxwell Lipford Hutchinson

  This file is part of HOMB.

  HOMB is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  HOMB is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with HOMB.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "homb_c.h"
#include "functions_c.h"

// MPI transfer directions and neighbours tags
#define IN -1
#define OUT 1
#define NORTH 0
#define SOUTH 1
#define WEST 2
#define EAST 3
#define BOTTOM 4
#define TOP 5

#ifdef USE_MPI
// MPI auxiliaries
// grid communicator
static MPI_Comm grid_comm;
// MPI tags
static const int ns_tag = 221, sn_tag=222, we_tag = 231, ew_tag=232, bt_tag = 241, tb_tag=242;
// neighbours ranks
static int ngb_n, ngb_s, ngb_w, ngb_e, ngb_b, ngb_t;
// transfer buffers
static Real *sides_buff, *sbuff_ns, *rbuff_ns, *sbuff_we, *rbuff_we, *sbuff_bt, *rbuff_bt;
// request array
static MPI_Request request[12];
#endif

// Global grid sizes, start-end indices for local grids,  MPI topology
int  ngxyz[3], sx, ex, sy, ey, sz, ez, nlx, nly, nlz, npxyz[3];

// OpenMP threads number
int nthreads;

// solution array
Real  *udata, *uOld, *uNew;

static int dims[3], coords[3];

// eigenvalue k vector (mode)
Real kx=1.0, ky=1.0, kz=1.0;

// Output control
static int vOut, pHeader;


/*********** FUNCTIONS ***********/
#ifdef USE_MPI
  void post_recv(void);
  void exchange_halos(void);
  void buffer_halo_transfers(int dir, double *norm, int update);
  void transfer_data(const int dir, int side);
#endif
void compute_local_grid_ranges( void );



#ifdef USE_MPI
void post_recv(){
  
  int npoints;

#pragma omp master
  {
    // N-S
    // receive ghost points for left face
    if ( coords[0] > 0 ) {
      npoints = (ey - sy + 1 ) * ( ez - sz + 1);
      MPI_Irecv(&rbuff_ns[0], npoints, REAL_MPI, ngb_n, 
		ns_tag, grid_comm, &request[0]);
    }
    else
      request[0] = MPI_REQUEST_NULL;
    
    
    // receive ghost points for right face
    if (coords[0] < dims[0]-1){
      npoints = (ey - sy + 1 ) * ( ez - sz + 1);
	MPI_Irecv(&rbuff_ns[npoints], npoints, REAL_MPI, ngb_s, 
		  sn_tag, grid_comm, &request[1]);
    }
    else
      request[1] = MPI_REQUEST_NULL;    
      
      // W-E recv
      // left face
      if ( coords[1] > 0 ) {
	npoints = (ex - sx + 1 ) * ( ez - sz + 1);
	MPI_Irecv(&rbuff_we[0], npoints, REAL_MPI, ngb_w,
		  we_tag, grid_comm, &request[2]);
      }
      else
	request[2] = MPI_REQUEST_NULL;

      // right face
      if (coords[1] < dims[1]-1) {
	npoints = (ex - sx + 1 ) * ( ez - sz + 1);
	MPI_Irecv(&rbuff_we[npoints], npoints, REAL_MPI, ngb_e,
		  ew_tag, grid_comm, &request[3]);
      }
      else
	request[3] = MPI_REQUEST_NULL;
      
       
      // B-T recv
      // left face
      if ( coords[2] > 0 ) {
	npoints =  (ex - sx + 1 ) * ( ey - sy + 1);
	MPI_Irecv(&rbuff_bt[0], npoints, REAL_MPI, ngb_b,
		  bt_tag, grid_comm, &request[4]);
      }
      else
	request[4] = MPI_REQUEST_NULL;

      // right face
      if (coords[2] < dims[2] - 1) {
	npoints =  (ex - sx + 1 ) * ( ey - sy + 1);
          MPI_Irecv(&rbuff_bt[npoints], npoints, REAL_MPI, ngb_t,
		    tb_tag, grid_comm, &request[5]);
      }
       else
	 request[5] = MPI_REQUEST_NULL;
  }
}


void exchange_halos(){

  MPI_Status status_sedrecv[12];
  int  npoints, ierr;
  // N-S send

#pragma omp master
  {
    // send to the right (i.e. s,e,t)
    if( coords[0] < dims[0] - 1) {
      npoints = (ey - sy + 1 ) * ( ez - sz + 1);
      MPI_Isend(&sbuff_ns[npoints], npoints, REAL_MPI, ngb_s, 
                ns_tag, grid_comm, &request[6]);
    }
    else
      request[6] = MPI_REQUEST_NULL;


    // send to the left
    if ( coords[0] > 0 ) {
      npoints = (ey - sy + 1 ) * ( ez - sz + 1);
      MPI_Isend(&sbuff_ns[0], npoints, MPI_REAL, ngb_n, 
                sn_tag, grid_comm, &request[7]);
    }
    else
      request[7] = MPI_REQUEST_NULL;

    // W-E send
    // send to the right (s,e,t)
    if( coords[1] < dims[1]-1) {
      npoints = (ex - sx + 1 ) * ( ez - sz + 1);
      MPI_Isend(&sbuff_we[npoints], npoints, REAL_MPI, ngb_e,
                we_tag, grid_comm, &request[8]);
    }
    else
      request[8] = MPI_REQUEST_NULL;
    
    // send to the left
    if ( coords[1] > 0 ) {
      npoints = (ex - sx + 1 ) * ( ez - sz + 1);
      MPI_Isend(&sbuff_we[0], npoints, REAL_MPI, ngb_w, 
		ew_tag, grid_comm, &request[9]);
    }
    else
      request[9] = MPI_REQUEST_NULL;
    
    // B-T send
    // send to the right (s,e,t)
    if( coords[2] < dims[2] - 1) {
      npoints = (ex - sx + 1 ) * ( ey - sy + 1);
      MPI_Isend(&sbuff_bt[npoints], npoints, REAL_MPI, ngb_t, 
                bt_tag, grid_comm, &request[10]);
    }
    else
      request[10] = MPI_REQUEST_NULL;
    
    // send to the left
    if ( coords[2] > 0 ) {
      npoints = (ex - sx + 1 ) * ( ey - sy + 1);
      MPI_Isend(&sbuff_bt[0], npoints, REAL_MPI, ngb_b, 
		     tb_tag, grid_comm, &request[11]);
    }
    else
      request[11] = MPI_REQUEST_NULL;
    
    ierr = MPI_Waitall(12, request, status_sedrecv);
    if ( ierr != MPI_SUCCESS) 
      fprintf(stderr, "error smoother waitall %d %d %d %d \n", coords[0], coords[1], coords[2], ierr);

  }      
}


void buffer_halo_transfers(int dir, Real *norm, int update){
  
  // fill the transfer buffers (dir > 0)  or halos ( dir < 0)
  //N-S

#pragma omp master
  {
    if ( coords[0] > 0 ) transfer_data(dir, NORTH);
    if ( coords[0] < dims[0] - 1 )  transfer_data(dir, SOUTH);
    if ( coords[1] > 0 ) transfer_data(dir, WEST);
    if ( coords[1] < dims[1] - 1 )  transfer_data(dir, EAST);
    if ( coords[2] > 0 ) transfer_data(dir, BOTTOM);
    if ( coords[2] < dims[2] - 1 ) transfer_data(dir, TOP);			 
    
 
    // when data is received the outer shell of local  must be updated in case of cco
    
    if (update) {
      stencil_update(sx, sx, sy+1, ey-1, sz+1, ez-1, norm);
      stencil_update(ex, ex, sy+1, ey-1, sz+1, ez-1, norm);
      stencil_update(sx, ex, sy,   sy,   sz+1, ez-1, norm);
      stencil_update(sx, ex, ey,   ey,   sz+1, ez-1, norm);
      stencil_update(sx, ex, sy,   ey,   sz,   sz,   norm);
      stencil_update(sx, ex, sy,   ey,   ez,   ez,   norm);
    }
  }
}


void transfer_data(const int dir, int side){
  
  int ib, i, j, k, ijk, s1, e1, s2, e2, s3, e3;
  Real * buff;
  
  s1 = sx; e1 = ex; s2 = sy; e2 = ey; s3 = sz; e3 = ez;
  switch( side )
    {
    case (NORTH) : ib = 0;  
      if (dir > 0) {
	buff = &sbuff_ns[0];
	e1 = sx;}
      else {
	buff = &rbuff_ns[0];
	s1 = sx - 1 ; e1 = s1;}
      break;
    case (SOUTH) : ib = (ey - sy + 1) * (ez - sz + 1);
      if (dir > 0) {
	buff = &sbuff_ns[ib];
	s1 = ex;}
      else  {
	buff = &rbuff_ns[ib];
	s1 = ex + 1; e1 = s1;}
      break;
    case (WEST)  : ib = 0; 
      if (dir > 0) {
	buff = &sbuff_we[0]; 
	e2 = sy;}
      else {
	buff = &rbuff_we[0];
	s2 = sy-1; e2 = s2;}
      break;
    case (EAST)  : ib = (ex - sx + 1) * (ez - sz +1); 
      if (dir > 0) {
	buff = &sbuff_we[ib];
	s2 = ey;}
      else  {
	buff = &rbuff_we[ib];
	s2 = e2 + 1; e2 = s2;}
      break;
    case(BOTTOM) :  ib = 0; 
      if (dir > 0) {
	buff = &sbuff_bt[ib];
	  e3 = sz;}
      else {
	buff = &rbuff_bt[ib];
	s3 = sz-1; e3 = s3; }
      break;
    case(TOP) : ib = (ex - sx + 1) * (ey - sy + 1); 
      if (dir > 0) {
	buff = &sbuff_bt[ib];
	s3 = ez;}
      else {
	buff = &rbuff_bt[ib];
	s3 = ez + 1; e3 = s3;}
      break;
    }
  
  if ( dir > 0 ){
    ib = 0;
    for (k = s3; k <= e3; ++k)
      for (j = s2; j <= e2; ++j)
	for (i = s1; i <= e1; ++i){
	  ijk = uindex(i, j, k);
	  buff[ib] = uOld[ijk];
	  ++ib;
	}
  }
  else { 
    ib = 0;
    for (k = s3; k <= e3; ++k)
      for (j = s2; j <= e2; ++j)
	for (i = s1; i <= e1; ++i){
	  ijk = uindex(i, j, k);
	  uOld[ijk] = buff[ib];
	  ++ib;
	}	
  }
  
}

#endif


void initContext(int argc, char *argv[]){
 
  /* "Logicals" for file output */
  vOut = 0;  pHeader = 1; pContext = 0; testComputation = 0;
  
  /* Defaults */
  ngxyz[0] = 33; ngxyz[1] = 33; ngxyz[2] = 33; niter = 20;  nthreads = 1; nproc = 1;
  npxyz[0] = 1; npxyz[1] = 1; npxyz[2] = 1;
  kernel_key = COMMON_KERNEL;
  int i;

  /* Cycle through command line args */
  i = 0;
  while ( i < argc - 1 ){
    ++i;
    //printf(" arg %d %d %s \n", i, argc, argv[i]);
    /* Look for grid sizes */
    if ( strcmp("-ng", argv[i]) == 0 ){
      sscanf(argv[++i],"%d",&ngxyz[0]);
      sscanf(argv[++i],"%d",&ngxyz[1]);
      sscanf(argv[++i],"%d",&ngxyz[2]);
    }
    /* Look for MPI topology */  
    else if ( strcmp("-np", argv[i]) == 0 ){
      sscanf(argv[++i],"%d",&npxyz[0]);
      sscanf(argv[++i],"%d",&npxyz[1]);
      sscanf(argv[++i],"%d",&npxyz[2]);
    }
    /* Look for number number of iterations*/  
    else if ( strcmp("-niter", argv[i]) == 0 ){
      sscanf(argv[++i],"%d",&niter);
    }
    /* Look for verbouse output */
    else if ( strcmp("-v", argv[i]) == 0 ){
      vOut = 1;
    }
    /* Look for "No Header" option */
    else if ( strcmp("-nh",argv[i]) == 0 ){
      pHeader = 0;
    }
    /* Look for "Print Context" option */
    else if ( strcmp("-pc",argv[i]) == 0 ){
      pContext = 1;
    }
    /* Look for kernel to use */
    else if ( strcmp("-model", argv[i]) == 0 ){
      ++i;
      if (strcmp("common",argv[i]) == 0)
	kernel_key = COMMON_KERNEL;
      else if (strcmp("blocked",argv[i]) == 0)
	kernel_key = BLOCKED_KERNEL;
      else if (strcmp("cco",argv[i]) == 0)
	kernel_key = CCO_KERNEL;
      else{
	printf("Wrong model specifier %s, try -help\n", argv[i]);
#ifdef USE_MPI
	MPI_Abort(MPI_COMM_WORLD,-1);
#else
	exit(-1);
#endif
      }
      
    }
    /* Look for eigenvalue test */
    else if ( strcmp("-t",argv[i] ) == 0){
      testComputation = 1;
      //printf(" i %d \n",i); 
    }
    /* Look for "verbose" standard out */
    else if ( strcmp("-help",argv[i]) == 0 ){
      printf("Usage: %s [-ng grid-size-x grid-size-y grid-size-z ] \
[-np num-proc-x num-proc-y num-proc-z]  -niter num-iterations \
[-v] [-t] [-pc] [-use_cco] [-nh] [-help] \n", argv[0]);
#ifdef USE_MPI
      MPI_Finalize();
#endif
      exit(EXIT_SUCCESS);
    }
    else{
      /* basic test on flag values */
      if (argv[i][0] != '-' ){
	printf("Wrong option flag %s, try -help\n", argv[i]);
#ifdef USE_MPI
	MPI_Abort(MPI_COMM_WORLD,-1);
#else
	exit(-1);
#endif
      }
    }

  }
}

void setPEsParams() {

  int periods[3] = {0, 0, 0}; 

#ifdef USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if ( nproc != npxyz[0] * npxyz[1] * npxyz[2]){
    fprintf(stderr, "MPI topology sizes %d %d %d does not fit with nproc %d \n", npxyz[0], npxyz[1], npxyz[2], nproc);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
#else
  nproc = 1;
#endif

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
#else 
  myrank = 0;
#endif

  /* generate 3d topology */
#ifdef USE_MPI
  MPI_Cart_create(MPI_COMM_WORLD, 3, npxyz, periods , 0, &grid_comm);
#endif

  compute_local_grid_ranges();

  /* Find number of threads per task */
  nthreads = 1;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif
  //  #pragma omp parallel shared(nThreads)
  //{
  //  #pragma omp single
  //     nThreads = omp_get_num_threads();
  //}
}

void compute_local_grid_ranges( void ){
  
  int i, nl[3], local_shift[3], r, periods[3];

#ifdef USE_MPI
  MPI_Cart_get(grid_comm, 3, dims, periods, coords);
#else
  for ( i =0; i < 3; ++i){
    dims[i] = 1; periods[i] = 0; coords[i] = 0;
  }
#endif  

  for ( i = 0; i < 3; ++i)
    nl[i] = ngxyz[i]/dims[i];
  
  for (i=0; i < 3; ++i){
    r = ngxyz[i]%dims[i];
    if ( coords[i] < r ){
      nl[i] += 1;
      local_shift[i] = coords[i] * nl[i];
    }
    else{
      local_shift[i] = r * (nl[i] + 1) + (coords[i] - r) * nl[i];
    }
    
  }

  sx = local_shift[0];
  ex = sx + nl[0]  - 1;
  sy = local_shift[1];
  ey = sy + nl[1] - 1;
  sz = local_shift[2];
  ez = sz + nl[2] - 1;

  /* sx, ex, ... are internal points
     therefore we need to  shift them if the rank
     has sides on domain boundary */
  if ( coords[0] == 0          ) sx = sx + 1;
  if ( coords[0] == dims[0] -1 ) ex = ex - 1;
  if ( coords[1] == 0          ) sy = sy + 1;
  if ( coords[1] == dims[1] -1 ) ey = ey - 1;
  if ( coords[2] == 0          ) sz = sz + 1;
  if ( coords[2] == dims[2] -1 ) ez = ez - 1;

  nlx = ex - sx + 1 + 2; nly = ey - sy + 1 + 2; nlz = ez - sz +1 + 2; 

  //printf("debug %d %d %d %d %d %d %d %d %d \n", nlx, nly, nlz, sx, ex, sy, ey, sz, ez);
  
  if ( ex - sx  < 2 || ey -sy < 2 || ez - sz < 2 ){
    printf("local domain too small, please choose another MPI topology or grid sizes \n");
    printf(" rank = %d, coords %d %d %d \n", myrank, coords[0], coords[1], coords[2]);
  }


#ifdef USE_MPI
  /* get the nearest neighbors ranks */

  MPI_Cart_shift(grid_comm, 0, 1, &ngb_n, &ngb_s);
  if ( coords[0] == 0 ) ngb_n = MPI_PROC_NULL;
  if ( coords[0] == dims[0] -1 ) ngb_s = MPI_PROC_NULL;

  MPI_Cart_shift(grid_comm, 1, 1, &ngb_w, &ngb_e);
  if ( coords[1] == 0 ) ngb_w = MPI_PROC_NULL;
  if ( coords[1] == dims[1] -1 ) ngb_e = MPI_PROC_NULL;
  
   MPI_Cart_shift(grid_comm, 2, 1, &ngb_b, &ngb_t);
  if ( coords[2] == 0 ) ngb_b = MPI_PROC_NULL;
  if ( coords[2] == dims[2] -1 ) ngb_t = MPI_PROC_NULL;

  //printf(" neighbors %d %d %d %d %d %d %d\n",myrank, ngb_n, ngb_s, ngb_w, ngb_e, ngb_b, ngb_t);

  // set transfer buffers
  sides_buff = malloc(4*((ey - sy + 1) * ( ez - sz + 1) + (ex - sx + 1) * (ez - sz + 1) 
			 + (ex - sx + 1) * (ey -sy + 1)) * sizeof(Real));

  sbuff_ns = &sides_buff[0];
  rbuff_ns = &sides_buff[2 * (ey - sy + 1) * ( ez - sz + 1)];
  sbuff_we = &sides_buff[4 * (ey - sy + 1) * ( ez - sz + 1)];
  rbuff_we = &sides_buff[4 * (ey - sy + 1) * ( ez - sz + 1) + 2 * (ex - sx + 1) * (ez - sz + 1)];
  sbuff_bt = &sides_buff[4 * (ey - sy + 1) * ( ez - sz + 1) + 4 * (ex - sx + 1) * (ez - sz + 1)];
  rbuff_bt = &sides_buff[4 * (ey - sy + 1) * ( ez - sz + 1) + 4 * (ex - sx + 1) * (ez - sz + 1)+ 2 * (ex - sx + 1) * (ey -sy + 1)];
#endif 
}


void initial_field() {
 
  int i, j, k, ijk, n = 2*nlx*nly*nlz; 
  
  udata = malloc(n * sizeof(Real));

  uOld = &udata[n/2]; uNew = &udata[0];
  // first tocuh policy for numa nodes
#pragma omp parallel  for schedule(static) default (shared) private(i,j,k, ijk)
  for (k = sz-1; k <= ez+1; ++k)
    for (j = sy-1; j <= ey+1; ++j)
      for (i = sx-1; i <= ex+1; ++i){
	ijk = uindex(i,j,k);
	  uOld[ijk] = sin((PI * i * kx) / (ngxyz[0] - 1)) * sin((PI * j * ky) / (ngxyz[1] - 1)) * sin((PI * k * kz) / (ngxyz[2] - 1));
	  uNew[ijk]=0.0;
      }
  
}


void printContext(void){
  printf("Global grid sizes   : %d %d %d \n", ngxyz[0], ngxyz[1], ngxyz[2]);
#ifdef USE_MPI
  printf("MPI topology        : %d %d %d \n", npxyz[0], npxyz[1], npxyz[2]);
#endif
  printf("Number of iterations: %d \n", niter);
  if (pHeader) 
    printf("Summary Standard Ouput with Header\n");
  else
    printf("Summary Standard Output without Header\n");
  
  if (vOut)
    printf( "Verbose Output \n");
 
#ifdef USE_MPI 
  if ( use_cco )
    printf("Using computation-communication overlap \n");
#endif
}

void check_norm(int iter, double norm){
/* test ration of to consecutive norm agains the smoother eigenvalue for the chosen mode */

  Real ln, gn, r, eig;
  static Real prev_gn;

  if (iter == 0){
    prev_gn = 1.0;
    if (myrank == ROOT){
      printf("Correctness check\n");
      printf("iteration, norm ratio, deviation from eigenvalue\n");
    }
  }
  ln = norm;
#ifdef USE_MPI
  MPI_Reduce(&ln, &gn, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);
#else
  gn = ln;
#endif
  if ( myrank == ROOT){
    r = sqrt(gn/prev_gn);
    eig = (cos(PI*kx/(ngxyz[0]-1)) + cos(PI*ky/(ngxyz[1]-1)) + cos(PI*kz/(ngxyz[2]-1)))/3.0;
    if( iter > 0) 
      printf("%5d    %12.5e     %12.5e\n", iter, r, r - eig);
    prev_gn = gn;
  }
} 


void timeUpdate(double *times){

  /* Update Root's times matrix to include all times */

#ifdef MPI
  if ( myrank == ROOT )
    call MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, times, NITER, MPI_DOULBE, ROOT, MPI_COMM_WORLD);
  else
    call MPI_Gather(times, niter, MPI_DOUBLE, MPI_BOTTOM, 0, MPI_DATATYPE_NULL, ROOT, MPI_COMM_WORLD);
#endif
}
 

void statistics(double *times,  
                double *minTime, double *meanTime, double *maxTime,
                double *stdvTime, double *NstdvTime){

  int iPE, iter, shift, ii;
  
  /* eliminate possible startup baias if niter is large enough */
  shift = ( niter > 20 ? 5 : 0);
    

  /* Compute mean, max, min of times */
  ii = 0;
    for (iPE = 0; iPE < nproc; iPE++){
      for (iter = shift; iter < niter; iter++){
	*meanTime += times[ii];
	*maxTime = MAX(*maxTime, times[ii]);
	*minTime = MIN(*minTime, times[ii]);
	++ii;
      }
      ii += shift;
    }
    *meanTime = *meanTime / (double) (niter - shift) / (double) nproc;

  /* Compute standard deviation of times */
    ii = 0;
    for (iPE = 0; iPE < nproc; iPE++){
      for (iter = shift; iter < niter; iter++){
	*stdvTime += (times[ii] - *meanTime) *
	  (times[ii] - *meanTime);
	++ii;
      }
      ii += shift;
    }
  *stdvTime = sqrt(*stdvTime / ((niter - shift)* nproc-1.0));

  /* Normalized standard deviation (stdv / mean) */
  *NstdvTime = *stdvTime / *meanTime;
}


void stdoutIO( double *times,  
              double minTime, double meanTime, double maxTime, 
	       double NstdvTime, double norm){
  int iter, ii, i;

  if (pHeader){
    printf("# Last norm %22.15e\n",sqrt(norm));
#ifdef USE_MPI
    printf("#==================================================================================================================================#\n");
    printf("#\tNPx\tNPy\tNPz\tNThs\tNx\tNy\tNz\tNITER\tmeanTime \tmaxTime  \tminTime  \tNstdvTime  #\n");
    printf("#==================================================================================================================================#\n");
#else
    printf("#==========================================================================================================#\n");
    printf("#\tNThs\tNx\tNy\tNz\tNITER\tmeanTime \tmaxTime  \tminTime  \tNstdvTime  #\n");
    printf("#==========================================================================================================#\n");
#endif 
  }
#ifdef USE_MPI
  printf("\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\t%9.3e\n",
         npxyz[0], npxyz[1],npxyz[2],nthreads,ngxyz[0],ngxyz[1],ngxyz[2],niter, meanTime, maxTime, minTime, NstdvTime);
#else
 printf("\t%d\t%d\t%d\t%d\t%d\t%9.3e\t%9.3e\t%9.3e\t%9.3e\n",
         nthreads,ngxyz[0],ngxyz[1],ngxyz[2],niter, meanTime, maxTime, minTime, NstdvTime);
#endif
  
  /* Only if "Verbose Output" asked for */
  if (vOut){ 
    printf("\n"); printf("\n");
    printf("# Full Time Output (rows are times, cols are tasks)\n");
    ii = 0;
    for (iter = 0; iter < niter; iter++){ 
      for (i = 0; i < nproc; i++){
        printf("%e \t",times[ii]);
	++ii;
      }
      printf("\n");
    }
  }
}

double my_wtime(){
  
#ifdef USE_MPI
  return (MPI_Wtime());
#else
#ifdef _OPENMP
  return (omp_get_wtime());
#else
  // needs a C standard timer
  printf("my_wtime not defined !!!\n");
  return(0.0);
#endif
#endif
}


