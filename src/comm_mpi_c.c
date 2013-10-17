/* functions for MPI halo exchange

Lucian Anton July 2013

*/

// MPI transfer directions and neighbours tags

#include "homb_c.h"
#include "kernels_c.h"
#include "comm_mpi_c.h"

#define NORTH 0
#define SOUTH 1
#define WEST 2
#define EAST 3
#define BOTTOM 4
#define TOP 5

#ifdef USE_MPI
// MPI auxiliaries
// MPI tags
static const int ns_tag = 221, sn_tag=222, we_tag = 231, ew_tag=232, bt_tag = 241, tb_tag=242;
// neighbours ranks
static int ngb_n, ngb_s, ngb_w, ngb_e, ngb_b, ngb_t;
// transfer buffers
static Real *sides_buff, *sbuff_ns, *rbuff_ns, *sbuff_we, *rbuff_we, *sbuff_bt, *rbuff_bt;
// request array
static MPI_Request request[12];
#endif

// solution array
extern Real  *udata, *uOld, *uNew;


#ifdef USE_MPI
void transfer_data(const struct grid_info_t *g, const int dir, int side);

void post_recv(const struct grid_info_t *g){
  
  int npoints;
  int sx = g->sx, ex = g->ex;
  int sy = g->sy, ey = g->ey;
  int sz = g->sz, ez = g->ez;
  const int *coords = g->cp, *dims = g->np;
  MPI_Comm grid_comm = g->comm;

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


void post_send(const struct grid_info_t *g){

  MPI_Status status_sedrecv[12];
  int  npoints, ierr;
  // N-S send

  int sx = g->sx, ex = g->ex;
  int sy = g->sy, ey = g->ey;
  int sz = g->sz, ez = g->ez;
  const int *coords = g->cp, *dims = g->np;
  MPI_Comm grid_comm = g->comm;
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
      MPI_Isend(&sbuff_ns[0], npoints, REAL_MPI, ngb_n, 
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


void buffer_halo_transfers(const struct grid_info_t *g, int dir, int update){
  
  // fill the transfer buffers (dir > 0)  or halos ( dir < 0)
  //N-S
  const int *coords = g->cp, *dims = g->np;
#pragma omp master
  {
    if ( coords[0] > 0 ) transfer_data(g, dir, NORTH);
    if ( coords[0] < dims[0] - 1 )  transfer_data(g, dir, SOUTH);
    if ( coords[1] > 0 ) transfer_data(g, dir, WEST);
    if ( coords[1] < dims[1] - 1 )  transfer_data(g, dir, EAST);
    if ( coords[2] > 0 ) transfer_data(g, dir, BOTTOM);
    if ( coords[2] < dims[2] - 1 ) transfer_data(g, dir, TOP);			 
    
 
    // when data is received the outer shell of local  must be updated in case of cco
    
    if (update) {
      int sx = g->sx, ex = g->ex;
      int sy = g->sy, ey = g->ey;
      int sz = g->sz, ez = g->ez;
      stencil_update(g, sx, sx, sy+1, ey-1, sz+1, ez-1);
      stencil_update(g, ex, ex, sy+1, ey-1, sz+1, ez-1);
      stencil_update(g, sx, ex, sy,   sy,   sz+1, ez-1);
      stencil_update(g, sx, ex, ey,   ey,   sz+1, ez-1);
      stencil_update(g, sx, ex, sy,   ey,   sz,   sz);
      stencil_update(g, sx, ex, sy,   ey,   ez,   ez);
    }
  }
}


void transfer_data(const struct grid_info_t *g, const int dir, int side){
  
  int ib, i, j, k, ijk, s1, e1, s2, e2, s3, e3;
  Real * buff;
  
  int sx = g->sx, ex = g->ex;
  int sy = g->sy, ey = g->ey;
  int sz = g->sz, ez = g->ez;

  s1 = sx; e1 = ex; s2 = sy; e2 = ey; s3 = sz; e3 = ez;
  switch( side )
    {
    case (NORTH) : ib = 0;  
      if (dir == OUT) {
	buff = &sbuff_ns[0];
	e1 = sx;}
      else {
	buff = &rbuff_ns[0];
	s1 = sx - 1 ; e1 = s1;}
      break;
    case (SOUTH) : ib = (ey - sy + 1) * (ez - sz + 1);
      if (dir == OUT) {
	buff = &sbuff_ns[ib];
	s1 = ex;}
      else  {
	buff = &rbuff_ns[ib];
	s1 = ex + 1; e1 = s1;}
      break;
    case (WEST)  : ib = 0; 
      if (dir == OUT) {
	buff = &sbuff_we[0]; 
	e2 = sy;}
      else {
	buff = &rbuff_we[0];
	s2 = sy-1; e2 = s2;}
      break;
    case (EAST)  : ib = (ex - sx + 1) * (ez - sz +1); 
      if (dir == OUT) {
	buff = &sbuff_we[ib];
	s2 = ey;}
      else  {
	buff = &rbuff_we[ib];
	s2 = e2 + 1; e2 = s2;}
      break;
    case(BOTTOM) :  ib = 0; 
      if (dir == OUT) {
	buff = &sbuff_bt[ib];
	  e3 = sz;}
      else {
	buff = &rbuff_bt[ib];
	s3 = sz-1; e3 = s3; }
      break;
    case(TOP) : ib = (ex - sx + 1) * (ey - sy + 1); 
      if (dir == OUT) {
	buff = &sbuff_bt[ib];
	s3 = ez;}
      else {
	buff = &rbuff_bt[ib];
	s3 = ez + 1; e3 = s3;}
      break;
    }
  
  if ( dir == OUT ){
    ib = 0;
    for (k = s3; k <= e3; ++k)
      for (j = s2; j <= e2; ++j)
	for (i = s1; i <= e1; ++i){
	  ijk = uindex(g, i, j, k);
	  buff[ib] = uOld[ijk];
	  ++ib;
	  //printf("debug transfer out: %d %g \n", myrank, uOld[ijk]);
	}
  }
  else { 
    ib = 0;
    for (k = s3; k <= e3; ++k)
      for (j = s2; j <= e2; ++j)
	for (i = s1; i <= e1; ++i){
	  ijk = uindex(g, i, j, k);
	  uOld[ijk] = buff[ib];
	  ++ib;
	  //printf("debug transfer in: %d %d %d %d %d %g \n", myrank, i,j,k, ijk, uOld[ijk]);
	}	
  }
  
}

void exchange_halos(const struct grid_info_t *g){
  post_recv(g);
  buffer_halo_transfers(g, OUT, NO_UPDATE);
  post_send(g);
  buffer_halo_transfers(g, IN, NO_UPDATE);
}

#endif


void compute_local_grid_ranges( struct grid_info_t * g ){
  
  int i, nl[3], local_shift[3], r, periods[3];
  int sx, ex, sy, ey, sz, ez;

#ifdef USE_MPI
  MPI_Cart_get(g->comm, 3, g->np, periods, g->cp);
#else
  for ( i =0; i < 3; ++i){
    g->np[i] = 1; periods[i] = 0; g->cp[i] = 0;
  }
#endif  

  for ( i = 0; i < 3; ++i)
    nl[i] = g->ng[i] / g->np[i];
  
  for (i=0; i < 3; ++i){
    r = g->ng[i] % g->np[i];
    if ( g->cp[i] < r ){
      nl[i] += 1;
      local_shift[i] = g->cp[i] * nl[i];
    }
    else{
      local_shift[i] = r * (nl[i] + 1) + (g->cp[i] - r) * nl[i];
    }
    
  }

  sx = local_shift[0];

  ex = sx + nl[0] - 1;
  sy = local_shift[1];
  ey = sy + nl[1] - 1;
  sz = local_shift[2];
  ez = sz + nl[2] - 1;

  /* sx, ex, ... are internal points
     therefore we need to  shift them if the rank
     has sides on domain boundary */
  if ( g->cp[0] == 0           ) sx = sx + 1;
  if ( g->cp[0] == g->np[0] -1 ) ex = ex - 1;
  if ( g->cp[1] == 0           ) sy = sy + 1;
  if ( g->cp[1] == g->np[1] -1 ) ey = ey - 1;
  if ( g->cp[2] == 0           ) sz = sz + 1;
  if ( g->cp[2] == g->np[2] -1 ) ez = ez - 1;

  g->sx = sx; g->ex = ex; g->sy = sy; g->ey = ey; g->sz = sz; g->ez = ez;
  g->nlx = ex - sx + 1 + 2; g->nly = ey - sy + 1 + 2; g->nlz = ez - sz + 1 + 2; 

  //printf("debug %d %d %d %d %d %d %d %d %d \n", g->nlx, g->nly, g->nlz, sx, ex, sy, ey, sz, ez);
  
  if ( ex - sx  < 2 || ey -sy < 2 || ez - sz < 2 ){
    printf("local domain too small, please choose another MPI topology or grid sizes \n");
    printf(" rank = %d, coords %d %d %d \n", g->myrank, g->cp[0], g->cp[1], g->cp[2]);
  }


#ifdef USE_MPI
  /* get the nearest neighbors ranks */
  MPI_Comm grid_comm = g->comm;
  MPI_Cart_shift(grid_comm, 0, 1, &ngb_n, &ngb_s);
  if ( g->cp[0] == 0 )           ngb_n = MPI_PROC_NULL;
  if ( g->cp[0] == g->np[0] - 1) ngb_s = MPI_PROC_NULL;

  MPI_Cart_shift(grid_comm, 1, 1, &ngb_w, &ngb_e);
  if ( g->cp[1] == 0 )           ngb_w = MPI_PROC_NULL;
  if ( g->cp[1] == g->np[1] - 1) ngb_e = MPI_PROC_NULL;
  
   MPI_Cart_shift(grid_comm, 2, 1, &ngb_b, &ngb_t);
  if ( g->cp[2] == 0 )           ngb_b = MPI_PROC_NULL;
  if ( g->cp[2] == g->np[2] - 1) ngb_t = MPI_PROC_NULL;

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


