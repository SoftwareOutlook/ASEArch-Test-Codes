/* kernels headers
 
Lucian Anton
July 2013

Copyright (c) 2014, Science & Technology Facilities Council, UK
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies, 
either expressed or implied, of the FreeBSD Project.


*/

//oid Gold_laplace3d_f(int NX, int NY, int NZ, int nxShift, Real* u1, Real* u2);
int uindex(const struct grid_info_t *grid, const int i, const int j, const int k);
void stencil_update(const struct grid_info_t *grid, int s1, int e1, int s2, int e2, int s3, int e3);
void laplace3d(const struct grid_info_t *g, double *tstart, double *tend);
//#ifdef USE_FORTRAN
void Gold_laplace3d_f(int *NX, int *NY, int *NZ, int *nxshift, Real* u1, Real* u2);
void Titanium_laplace3d_f(int *NX, int *NY, int *NZ, int *nxshift, Real* u1, Real* u2);
void Blocked_laplace3d_f(int *NX, int *NY, int *NZ, int *BX, int *BY, int *BZ, int *nxshift, Real* u1, Real* u2);
//void Wave_laplace3d_f(int *NX, int *NY, int *NZ, int *nxshift, int *BX, int *BY, int *BZ, int *iter_block, int *nthreads, const int *nthreads_per_column, Real *uOld, Real *uNew);
//void Wave_diagonal_laplace3d_f(int *NX, int *NY, int *NZ, int *nxshift, int *BX, int *BY, int *BZ, int *iter_block, Real* u1, Real* u2);
//#endif
