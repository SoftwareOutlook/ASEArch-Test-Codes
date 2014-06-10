/** @file
 *This file contains the single and double precision kernel calls
 *\todo include the header to grab the typedef of Real
 */

#include "../jacobi_c.h"
/** jacobi_relaxation_ocl --
 * @param Nx - The x size of the domain
 * @param Ny - The y size of the domain
 * @param Nz - The z size of the domain
 * @param d_u1 - The input array
 * @param d_u2 - The output array 
 */
__kernel void  jacobi_relaxation_ocl(const int Nx,
				     const int Ny,
				     const int Nz,
				     global const Real* restrict d_u1,
				     global Real* restrict d_u2){
  const Real sixth=1.0/6.0;

  //Thread Indices
  const int x =get_global_id(0);
  const int y =get_global_id(1);
  const int z =get_global_id(2);
  const int loc=z*Ny*Nx+y*Nx+x;
  
 
  if(x!=0&&x!=(Nx-1)&&y!=0&&y!=(Ny-1)&&z!=0&&z!=(Nz-1)){
    d_u2[loc]= sixth*(d_u1[loc-Ny*Nx]+d_u1[loc+Nx*Ny]+d_u1[loc-Nx]+d_u1[loc+Nx]+d_u1[loc-1]+d_u1[loc+1]);
    
  }
}
