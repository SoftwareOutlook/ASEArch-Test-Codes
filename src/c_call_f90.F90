module kernels

! #if _OPENACC
!   use openacc
! #endif
  use iso_c_binding
  implicit none

!   type, bind(C) :: grid_info_t
!   INTEGER(C_INT) :: ng(3), nb(3), threads_per_column, sx, ex, sy, ey, &
!        sz, ez, nlx, nly, nlz
!   INTEGER(C_INT) :: nproc, myrank, np(3), cp(3), cmod_key, alg_key, malign, nwaves
!   !integer(C_SIGNED_CHAR) :: i
!   !REAL(C_DOUBLE) :: d1
!   !COMPLEX(C_FLOAT_COMPLEX) :: c1
!   CHARACTER(KIND=C_CHAR) :: cmod_name(31+1), alg_name(31+1)

! end type grid_info_t
#if USE_DOUBLE_PRECISION
  integer, parameter :: wp = kind(0.0d0)
#endif
  !! ??≈
  integer, parameter :: wp = kind(4)
  !! real(wp), parameter :: pi = 4.0*atan(1.0)
  real(kind=4), parameter :: sixth = 1.0/6.0 


  

contains
    ! subroutine my_example(a, b) BIND(C, name="example")
    !   real(c_double), intent(in) :: b
    !   real(c_double), intent(inout) :: a
    !   a = a*b
    ! end subroutine

    
 ! subroutine initial_field_c(grid, uOld, uNew) BIND(C, name='initial_field_c_f')
!      implicit none
     
!      type(grid_info_t), intent(in) :: grid
!      real(c_double), dimension(grid%nlx*grid%nly*grid%nlz), intent(inout) :: uOld
!      real(c_double), dimension(grid%nlx*grid%nly*grid%nlz), intent(inout) :: uNew

!      !!real(wp) :: uOldval
!      integer i, j, k
!      integer ijk
!      !!integer n 
!     ! real(wp), dimension(:), allocatable :: uOld
!     ! real(wp), dimension(:), allocatable :: uNew
!     ! allocate (uOld((grid%nlx)*(grid%nly)*(grid%nlz)))
!     ! allocate (uNew((grid%nlx)*(grid%nly)*(grid%nlz)))
!      !allocate(uOld(10,10,10))
!      !!n = grid%nlx * grid%nly * grid%nlz
     
!      !!
!      !!pi 
!      !!real, parameter :: inv6 = 1.0/6.0
!      !INTEGER(c_size_t)
!     !s =  (size_t)n * sizeof(Real);
!     !udata = malloc(s);
!     !//printf("no maling %d \n", n*sizeof(Real));
!   !}

!      !uOld = &udata[n/2]; uNew = &udata[0];
 
!      !allocate ( uOld (grid%sx-1:grid%ex+1, grid%sy-1:grid%ey+1, grid%sz-1:grid%ez+1) )
!      !allocate ( uNew (grid%sx-1:grid%ex+1, grid%sy-1:grid%ey+1, grid%sz-1:grid%ez+1) )
   
!      ! first touch
! !!! !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k) SCHEDULE(STATIC) 
!      do k = grid%sx-1, grid%ex+1
!          do j = grid%sy-1, grid%ey+1
!             do i = grid%sx-1, grid%ex+1
!                ijk = (i - (grid%sx - 1) + (j - (grid%sy -1)) * grid%nlx + (k - (grid%sz - 1)) * grid%nlx * grid%nlz)
!                if ( ( i == grid%sx-1 .or. i == grid%ex+1 ) .or. &
!                    ( j == grid%sy-1 .or. j == grid%ey+1 ) .or. &
!                    ( k == grid%sz-1 .or. k == grid%ez+1 ) ) then 
!                   uOld(ijk) = 0.0_c_double
!                  !grid%cmod_key = 1
!                else
!                 !grid%cmod_key = 0
!                 !print *, grid%cmod_key
! 	! uOld[ijk] = sin((PI * i * kx) / (g->ng[0] - 1)) * sin((PI * j * ky) / (g->ng[1] - 1)) * sin((PI * k * kz) / (g->ng[2] - 1));
! 	! uNew[ijk]=0.0;

!                   uNew(ijk) = 0.0_c_double
!                   uOld(ijk) = sin((pi * i) / real(grid%ng(1) - 1, wp)) * &
!                        sin((pi * j) / real(grid%ng(2) - 1, wp)) * &
!                        sin((pi * k) / real(grid%ng(3) - 1, wp))
!                   !uOld(ijk) = uOldval_c_double
! !                  uOld(ijk) = (pi*1.0*(i-1)/real(grid%nlx-1, wp)) * &
!  !                                 (pi*1.0*(j-1)/real(grid%nly-1, wp)) * &
!   !                               (pi*1.0*(k-1)/real(grid%nlz-1, wp))
!                endif
!             enddo
!          enddo
!      enddo
! !      !return uOld uNew
! ! !!!  !$OMP END PARALLEL DO
! !     return uOld

 
!    end subroutine initial_field_c

   subroutine Gold_laplace3d_f(nx, ny, nz, nxShift, u1, u2) BIND(C, name="Gold_laplace3d_f")
     use iso_c_binding
     !!type(grid_info_t), intent(in) :: grid
     integer(c_int), intent(in) :: nx, ny , nz, nxShift
     real(c_float), intent(in), dimension(0:(nx*ny*nz)-1) :: u1
     real(c_float), intent(inout), dimension(0:(nx*ny*nz)-1) :: u2
     
     !type(c_ptr), intent(inout) :: u2
     !integer(c_int), intent(in) :: NX, NY, NZ, nxShift
     integer i, j, k
     integer ijk     !!real(c_double), dimension(
     !integer(c_int), pointer :: c_int_nx, c_int_ny, c_int_nz, c_int_nxshift
     !real(c_float), pointer :: c_array_u1(:)
     !real(c_float), pointer :: c_array_u2(:)
     !call c_f_pointer(nxShift, c_int_nxshift)
     !call c_f_pointer(nx, c_int_nx)
     !call c_f_pointer(ny, c_int_ny)
     !call c_f_pointer(nz, c_int_nz)
     !print *, "u1(1500:1510)", u1(1500:1510)
     !print *, "F", nx, ny, nz
     !print *, "test" 
     !call c_f_pointer(u1, c_array_u1, (/c_int_nx*c_int_ny*c_int_nz/) )
     !call c_f_pointer(u2, c_array_u2, (/c_int_nx*c_int_ny*c_int_nz/) )
     !real(c_float), dimension(NX*NY*NZ), intent(in) :: u1
     !real(c_float), dimension(NX*NY*NZ), intent(inout) :: u2

     !!real(wp) :: uOldval

#if _OPENACC
!$ACC data deviceptr(u1, u2)
!$ACC kernels loop collapse(3) independent private(i,j,k,ind)
#else     
!$OMP parallel DO schedule(static) default(none) shared(u1, u2, NX, NY, NZ, nxShift) private(i,j,k,ijk)    
#endif
     do k = 0, NZ-1
         do j = 0, NY-1
            do i = 0, NX-1
               ! add 1 for Fortran indexing
               !ijk = 1 + i  + j*nxShift + k*nxShift*NY
               ijk = i + j*nxshift + k*nxshift*ny
               if ( i == 0 .or. i == NX-1 .or. &
                    j == 0 .or. j == NY-1 .or. &
                    k == 0 .or. k == NZ-1  ) then 
                  u2(ijk) = u1(ijk)
               else
                  u2(ijk) = (u1(ijk - 1) + u1(ijk + 1) + &
                       u1(ijk - nxShift) + u1(ijk + nxShift) +  &
                       u1(ijk - nxShift*NY) + u1(ijk + nxShift*NY )) * sixth
               endif
            enddo
         enddo
      enddo
#if _OPENACC
!$acc end kernels
!$acc end data
#else     
!$OMP END PARALLEL DO
#endif
      
    end subroutine Gold_laplace3d_f
    
    subroutine vec_oneD_loop(nx,ny,nz, n, uNorth, uSouth, uWest, uEast, uBottom, uTop, w)
      integer, intent(in) :: n, nx, ny, nz
      real(c_float), dimension(0:(nx*ny*nz)-1), intent(in) :: uNorth, uSouth, uWest, uEast, uBottom, uTop
      real(c_float), dimension(0:(nx*ny*nz)-1), intent(inout) :: w
      integer i
      
#if 0
#if __INTEL_COMPILER
!dir$ ivdep
#endif
#if __IBMC__
!$IBM* INDEPENDENT
#endif
#endif

#if __INTEL_COMPILER
!dir$ ivdep
#endif
      do i=0, n-1
         w(i) = sixth * (unorth(i) + usouth(i) + uwest(i) + ueast(i) + ubottom(i) + utop(i))
      end do
         
    end subroutine vec_oneD_loop
    
    subroutine Titanium_laplace3d(nx, ny, nz, nxshift, u1, u2) BIND(C, name="Titanium_laplace3d_f")
      integer(c_int), intent(in) :: nx, ny, nz, nxshift
      real(c_float), intent(in), dimension(0:(NX*NY*NZ)-1) :: u1
      real(c_float), intent(inout), dimension(0:(NX*NY*NZ)-1) :: u2
      !integer   nxy = nxshift * NY
      integer   i, j, k, ind, indmj, indpj, indmk, indpk, nxy
      nxy = nxshift * ny
      !print *, "F", nx, ny, nz
      !!print *, "u1(1500:1510)", u1(1500:1510)
      !!print *, "u2(1500:1510)", u2(1500:1510)
      !print *, "u1(1000)", u1()
!$OMP parallel default(none) shared(u1,u2,nx,ny,nz,nxy,nxshift) private(i,j,k,ind,indmj,indpj,indmk,indpk)
!$OMP do schedule(static) collapse(2)
      do k=1, nz-2
         do j=1, ny-2
            ind = j*nxshift + k*nxy
            indmj = ind - nxshift
            indpj = ind + nxshift
            indmk = ind - nxy
            indpk = ind + nxy
            ! if (k .eq. 1) then
            !    print *, "k", k
            !    print *, "j", j
            !    print *, "ind", ind
            !    print *, "indmj", indmj
            !    print *, "indpj", indpj
            !    print *, "indmk", indmk
            !    print *, "indpk", indpk
            ! end if
            
#if USE_VEC1D
           call vec_oneD_loop(nx,ny,nz, nx - 2, u1(ind), u1(ind+2), u1(indmj+1), u1(indpj+1), u1(indmk+1), u1(indpk+1), u2(ind+1))
#else
            do i=1, nx-2
               ! if (i .eq. 1) then
               !    print *, "ind + 1", ind + 1
               !    print *, "ind + i - 1", ind + i - 1
               !    print *, "ind + i + 1", ind + i + 1
               !    print *, "indmj + i", indmj + i
               !    print *, "indpj + i", indpj + i
               !    print *, "indmk + i", indmk + i
               !    print *, "indpk + i", indpk + i
               ! end if
               
               u2(ind + i) = (u1(ind + i - 1) + u1(ind + i +1) &
                    + u1(indmj + i) + u1(indpj + i) &
                    + u1(indmk + i) + u1(indpk + i) ) * sixth
            end do
#endif           
         end do
      end do
      
      !!print *, "u1(1500:1510)", u1(1500:1510)
      !!print *, "u2(1500:1510)", u2(1500:1510)
!$OMP end do
!$OMP end parallel
      
    end subroutine Titanium_laplace3d

    subroutine Blocked_laplace3d(nx, ny, nz, bx, by, bz, nxshift, u1, u2) BIND(C, name="Blocked_laplace3d_f")
      
      integer(c_int), intent(in) :: nx, ny, nz, bx, by, bz, nxshift
      real(c_float), dimension(nx*ny*nz), intent(in) :: u1
      real(c_float), dimension(nx*ny*nz), intent(inout) :: u2
      integer i, j, k, ind, indmj, indpj, indmk, indpk, nxy, ii, jj, kk

      nxy = nxshift * ny
!$OMP parallel default(none) shared(u1,u2,nx,ny,nz,nxy,nxshift,bx,by,bz) private(i,j,k,ind,indmj,indpj,indmk,indpk,ii,jj,kk)
!$OMP do schedule(static,1) collapse(3)
      do kk=1, nz-2
         do jj=1, ny-2
            do ii=1, nx-2
               do k = kk, min(1 + kk + bz, nz-2)
                  do j = jj, min(1 + jj + by, ny-2)
                     ind = j*nxshift + k*nxy
                     indmj = ind - nxshift
                     indpj = ind + nxshift
                     indmk = ind - nxy
                     indpk = ind + nxy
#if USE_VEC1D
                     call vec_oneD_loop(nx,ny,nz, min(ii+bx, nx-1)-ii, u1(ind+ii-1), u1(ind+ii+1), &
                          u1(indmj+ii), u1(indpj+ii), u1(indmk+ii), u1(indpk+ii), u2(ind+ii))
#else
                     do i=ii, min(1 + ii + bx, nx-2)
                        u2(ind + i) = ( u1(ind + i - 1) + u1(ind + 1 + 1) &
                             + u1(indmj + i) + u1(indpj + 1) &
                             + u1(indmk + i) + u1(indpk + 1) ) * sixth        
                     end do
#endif                     
                  end do
               end do
            end do
         end do
      end do
!$OMP end do
!$OMP end parallel
    end subroutine Blocked_laplace3d
    

!     subroutine Wave_laplace3d(nx, ny, nz, nxshift, bx, by, bz, iter_block, nthreads, &
!          threads_per_column, u1, u2) BIND(C, name="Wave_laplace3d_f")
!       integer(c_int), intent(in) :: nx, ny, nz, bx, by, bz, nxshift, iter_block, nthreads, &
!            threads_per_column
!       real(c_float), dimension(nx*ny*nz), intent(in) :: u1
!       real(c_float), dimension(nx*ny*nz), intent(inout) :: u2
!       integer ind, indmj, indpj, indmk, indpk, i, j, k, jj, kk
!       integer jblock, kblock
!       integer iplane, left_y, right_y, nwaves
!       real, dimension(:), allocatable :: unew, uold
!       integer nby, nbz, jth, kth
      
!       integer max_thread_column
!       integer nxy
!       integer OMP_GET_THREAD_NUM
!       nxy = nxshift * ny
!       max_thread_column = nthreads/threads_per_column
! !$OMP parallel shared(u1, u2, NX, NY, NZ, NXY, nxShift, BX, BY, BZ, iter_block, max_thread_column, threads_per_column)
!       nwaves = iter_block
!       allocate(unew(nx*ny*nz))
!       allocate(uold(nx*ny*nz))
!       nby = (ny - 2) / by
!       if (mod ((ny - 2), by) .NE. 0) then
!          nby = nby + 1
!       end if
!       if (mod ((nz - 2), bz) .NE. 0) then
!          nbz = nbz + 1
!       end if

!       ! the threads are partitioned in threads_per_column, max_threads_column columns
!       ! k is the fast direction
! #if _OPENMP
!       jth = OMP_GET_THREAD_NUM()/threads_per_column
!       kth = mod(OMP_GET_THREAD_NUM(), threads_per_column)
! #else
!       jth = 0
!       kth = 0
! #endif
!       ! loop over the grid blocks in diagonal planes
!       do iplane = 0, (nby - 1 + nbz -1) + 2 * (nwaves -1)

!          !set the left and right limits for wave
!          !tricky here, it's easy to lose some tiles
!          !add more explanation

!          if (iplane - 2 * (nwaves - 1) < nbz) then
!             left_y = 0
!          else
!             left_y = iplane - 2 * (nwaves - 1) - nbz + 1
!          end if

!          if (iplane < nby) then
!             right_y = iplane
!          else
!             right_y = nby
!          end if

!          ! over the blocks belonging to the waves at a given iplane
!          ! first wave has the iplane index, subsequent ones are behind with stride 2
!          do jblock = left_y + jth, right_y, max_thread_column
!             do kblock = iplane - jblock - 2 * kth, max(0, iplane - jblock - 2 * (nwaves-1)), -2 * threads_per_column
!                !some blocks fall outside grid
!                if (kblock .LE. nbz - 1) then
!                   if (jblock .LE. nby - 1) then
!                      !where to write the new values; get the index of the wave
!                      if ( mod((iplane - jblock - kblock)/2, 2) .eq. 0) then
!                         unew = u2
!                         uold = u1
!                      else
!                         unew = u1
!                         uold = u2
!                      end if
!                      jj = jblock * by + 1
!                      kk = kblock * bz + 1
!                      do k = kk, min(kk + bz + 1, nz-2)
!                         do j = jj, min(jj + by + 1, nz -2)
!                            ind = j*nxshift + k*nxy
!                            indmj = ind - nxshift
!                            indpj = ind + nxshift
!                            indmk = ind - nxy
!                            indpk = ind + nxy
! #if USE_VEC1D
!                            call vec_oneD_loop(nx,ny,nz, nx - 2, u1(ind), u1(ind+2), u1(indmj+1), &
!                                 u1(indpj+1), u1(indmk+1), u1(indpk+1), u2(ind+1))
! #else
!                            do i=1, nx-2
!                               unew(ind + 1) = (uold(ind + i - 1) + uold(ind + i +1) &
!                                    + uold(indmj + i) + uold(indpj + i) &
!                                    + uold(indmk + i) + uold(indpk + i) ) * sixth
!                            end do
! #endif
!                         end do
!                      end do
!                   end if
!                end if
!             end do
!          end do
! !$OMP barrier
!       end do
! !$OMP end parallel
!     end subroutine Wave_laplace3d

!     subroutine Wave_diagonal_laplace3d(nx, ny, nz, nxshift, bx, by, bz, iter_block, u1, u2) &
!          BIND(C, name="Wave_diagonal_laplace3d_f")
!       integer(c_int), intent(in) :: nx, ny, nz, bx, by, bz, nxshift, iter_block
!       real(c_float), dimension(nx*ny*nz), intent(in) :: u1
!       real(c_float), dimension(nx*ny*nz), intent(inout) :: u2
!       integer nxy, ind, indmj, indpj, indmk, indpk, ii, jj, kk, i, j, k
!       integer iplane, wplane, iwave, left_y, right_y, iblock, nwaves, nby, nbz
!       real(kind=4), allocatable, dimension(:) :: unew, uold, tmp
! !$OMP parallel shared(u1, u2, nx, ny, nz, nxy, nxshift, bx, by, bz, iter_block)
!       allocate(unew(nx*ny*nz))
!       allocate(uold(nx*ny*nz))
!       allocate(tmp(nx*ny*nz))
!       nxy = nxshift * ny
!       nwaves = iter_block
!       nby = (ny - 2) / by
!       nbz = (nz - 2) / bz
!       if (mod( ny - 2, by) .NE. 0) then
!          nby = nby + 1
!       end if
!       if (mod( nz - 2, bz) .NE. 0) then
!          nbz = nbz + 1
!       end if
!       !iplane = 0
!       do iplane = 0, (nby -  1 + nbz + 1) + 2 * (nwaves - 1)
!          !iplane = iplane + 1
!          !if ( iplane .GT. (nby - 1 + nbz + 1) + 2 * (nwaves -1) ) exit
!          unew = u2
!          uold = u1
!          iwave = 0
       
!          do iwave = 0, nwaves - 1
!             !iwave = iwave + 1
!             !if ( iwave .GE. nwaves) exit
!             wplane = iplane - 2 * iwave

!             if (wplane .GE. 0) then
!                if (wplane .LE. (nby - 1 + nbz - 1)) then
!                   if (wplane .LT. nbz) then
!                      left_y = 0
!                   else
!                      left_y = wplane - nbz + 1
!                   end if

!                   if (wplane .LT. nby) then
!                      right_y = wplane
!                   else
!                      right_y = nby - 1
!                   end if
!                   !iblock = left_y
! !$OMP do schedule(dynamic,1)
!                   do iblock = left_y, right_y - 1
!   !                   iblock = iblock + 1
!        !              if (iblock .GE. right_y)
!       !               exit !$OMP end do nowait                     
!                      jj = iblock * by + 1
!                      kk = (wplane - iblock) * bz + 1
!                      do k = kk, min(kk + bz + 1, nz - 2)
!                         do j = jj, min(jj + by + 1, ny - 2)
!                            ind = j*nxshift + k*nxy
!                            indmj = ind - nxshift
!                            indpj = ind + nxshift
!                            indmk = ind - nxy
!                            indpk = ind + nxy
! #if USE_VEC1D
!                            call vec_oneD_loop(nx,ny,nz, nx - 2, u1(ind), u1(ind+2), u1(indmj+1), &
!                                 u1(indpj+1), u1(indmk+1), u1(indpk+1), u2(ind+1))
! #else
!                            do i=1, nx-2
!                               unew(ind + 1) = (uold(ind + i - 1) + uold(ind + i +1) &
!                                    + uold(indmj + i) + uold(indpj + i) &
!                                    + uold(indmk + i) + uold(indpk + i) ) * sixth
!                            end do
! #endif
!                         end do
!                      end do
!                   end do
! !$OMP end do nowait                 
!                end if
!             end if
!             tmp = unew
!             unew = uold
!             uold = tmp
!          end do         
! !$OMP barrier
!       end do
! !$OMP END PARALLEL
!     end subroutine Wave_diagonal_laplace3d

                
  end module kernels
  
