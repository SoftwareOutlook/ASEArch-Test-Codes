!
! DL_HOMB fortran version, see functions_c.c and homb_c.c for more details
!
! Lucian Anton 8/07/2013

module functions
  implicit none
  
! Parameters

  integer, parameter :: wp = kind(0.d0)
  real(wp), parameter :: pi = 4.0_wp*atan(1.0_wp), inv6 =1.0_wp/6.0_wp
  integer, parameter :: ROOT = 0

  ! MPI topology neigborhs: north, south, west, east, bottom, top
  integer, private :: ngb_n, ngb_s, ngb_w, ngb_e, ngb_b, ngb_t
  ! MPI tags for data exchages
  integer, private, parameter :: ns_tag = 221, sn_tag=222, &
                        we_tag = 231, ew_tag=232, &
                        bt_tag = 241, tb_tag=242
! MPI topology 
  integer, private :: grid_comm, dims(3), coords(3)

! Global variables

! Global grid sizes, MPI topology dims, Iterations (samples)
  integer  ngxyz(3), npxyz(3),&
           sx, ex, sy, ey, sz, ez, &
           niter

! eigenvalue modes
  real(wp) :: kx = 1.0_wp, ky = 1.0_wp, kz = 1.0_wp

! Input/Ouput logicals 
logical vOut, test_computation, pHeader, pContext 

! Number of Tasks, Threads, rows in local section of distributed grid  

! MPI task ID 
integer myrank, nproc, nthreads

! solution array
real(wp), allocatable :: u(:,:,:,:)

! Computation communication overlap
logical use_cco

contains

   subroutine jacobi_smoother(iteration, norm)
     use mpi
     implicit none
     
     integer, intent(in)   :: iteration
     real(wp), intent(out) :: norm
     
! Locals
     integer i, j, k, old, new, request(12)
     real(wp) w, buff_ns(sy:ey, sz:ez, 2), &
                 rbuff_ns(sy:ey, sz:ez, 2), &
                 buff_we(sx:ex, sz:ez, 2), &
                 rbuff_we(sx:ex, sz:ez, 2), &
                 buff_bt(sx:ex, sy:ey, 2), &
                 rbuff_bt(sx:ex, sy:ey, 2)

     new = 3 - mod(iteration,2) - 1 ! convention: new is 1 at iteration 1 
     old = 3 - new
     norm = 0.0_wp

!$OMP PARALLEL IF (nThreads > 1) DEFAULT(NONE) &
!$OMP SHARED(sx, ex, sy, ey, sz, ez, u, request, buff_ns, rbuff_ns, &
!$OMP buff_we, rbuff_we, buff_bt, rbuff_bt,old, new) & 
!$OMP PRIVATE (i, j, k, w)REDUCTION(+ : norm)

     call post_recv

     call  buffs_halos_transfers(1)

!$OMP BARRIER
     
     call exchange_halos
     
!$OMP BARRIER

     call buffs_halos_transfers(-1)

!$OMP BARRIER

!$OMP DO SCHEDULE(STATIC) COLLAPSE(2)
     do k = sz, ez
        do j = sy, ey
           do i = sx, ex
              w = inv6 * &
                   (u(i-1,j,k,old) + u(i+1,j,k,old) +&
                   u(i,j-1,k,old) + u(i,j+1,k,old) + &
                   u(i,j,k-1,old) + u(i,j,k+1,old))
              norm = norm + w*w
              !w = exp(w/(1.0_wp+w*w))
              u(i,j,k,new) = w
           enddo
        enddo
     enddo
!$OMP ENDDO NOWAIT
!$OMP END PARALLEL

   contains 

     subroutine post_recv
       implicit none

       integer npoints, ierr

!$OMP MASTER
       ! N-S
       ! receive ghost points for left face
       if ( coords(1) > 0 ) then
          ! Post receives the next color on the left face
          npoints = (ey - sy + 1 ) * ( ez - sz + 1)
          call MPI_Irecv(rbuff_ns(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_n, &
               ns_tag, grid_comm, request(1), ierr)
       else
          request(1) = MPI_REQUEST_NULL
       endif

        ! receive ghost points for left face
       if (coords(1) < dims(1)-1) then
          !  receiving from right
          npoints = (ey - sy + 1 ) * ( ez - sz + 1)
          call MPI_Irecv(rbuff_ns(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_s, &
               sn_tag, grid_comm, request(2), ierr)
       else
           ! with this values for unused request we can use mai_waitall at for communication finalisation
          request(2) = MPI_REQUEST_NULL
       endif

        ! W-E recv

       if ( coords(2) > 0 ) then
          ! Post receives for the next color on the left face
          npoints = (ex - sx + 1 ) * ( ez - sz + 1)
          !           write(0,*) 'recv on left ' , myid,icol, npoints
          call MPI_Irecv(rbuff_we(:,:,1),npoints, MPI_DOUBLE_PRECISION, ngb_w,&
               we_tag, grid_comm, request(3), ierr)
       else
          request(3) = MPI_REQUEST_NULL
       endif
       ! receive ghost points for left face
       if (coords(2) < dims(2)-1) then
          !  receiving from right
          npoints = (ex - sx + 1 ) * ( ez - sz + 1)
          !            write(0,*) 'recv on right ' , myid,icol, npoints
          call MPI_Irecv(rbuff_we(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_e,&
               ew_tag, grid_comm, request(4), ierr)
       else
          ! with this values for unused request we can use mai_waitall at for communication finalisation
          request(4) = MPI_REQUEST_NULL
       endif
       
       ! B-T recv
       
       if ( coords(3) > 0 ) then
          ! Post receives for the next color on the left face
          npoints =  (ex - sx + 1 ) * ( ey - sy + 1)
          call MPI_Irecv(rbuff_bt(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_b,&
               bt_tag, grid_comm, request(5), ierr)
       else
          request(5) = MPI_REQUEST_NULL
       endif
       ! receive ghost points for left face
       if (coords(3) < dims(3)-1) then
          !  receiving from right
          npoints =  (ex - sx + 1 ) * ( ey - sy + 1)
          call MPI_Irecv(rbuff_bt(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_t,&
               tb_tag, grid_comm, request(6), ierr)
       else
          ! with this values for unused request we can use mai_waitall at for communication finalisation
          request(6) = MPI_REQUEST_NULL
       endif
       
!$OMP END MASTER
     end subroutine post_recv


     subroutine buffs_halos_transfers(dir)
       implicit none
       integer, intent(in) :: dir !  1 grid -> buff
                                  ! -1 buff -> halo   
       
       ! fill the transfer buffers, for black the updateis done in update face
! N-S

!$OMP SINGLE
     if ( coords(1) > 0 ) then
        if ( dir > 0 ) then 
           buff_ns(sy:ey,sz:ez,1) = u(sx,sy:ey,sz:ez,old)
        else
           u(sx-1,sy:ey,sz:ez,old) = rbuff_ns(sy:ey,sz:ez,1)
        endif
     endif
!$OMP END SINGLE NOWAIT

!$OMP SINGLE
     if( coords(1) < dims(1)-1) then
        if ( dir > 0) then 
           buff_ns(sy:ey,sz:ez,2) = u(ex,sy:ey,sz:ez,old)
        else
            u(ex+1,sy:ey,sz:ez,old) = rbuff_ns(sy:ey,sz:ez,2)
         endif
     endif
!$OMP END SINGLE NOWAIT

! W-E

!$OMP SINGLE
     if( coords(2) > 0) then
        if ( dir > 0 ) then 
           buff_we(sx:ex,sz:ez,1) = u(sx:ex,sy,sz:ez,old)
        else
          u(sx:ex,sy-1,sz:ez,old) =rbuff_we(sx:ex,sz:ez,1)
       endif
     endif
!$OMP  END SINGLE NOWAIT

!$OMP SINGLE
     if( coords(2) < dims(2)-1) then
        if ( dir > 0 ) then 
           buff_we(sx:ex,sz:ez,2) = u(sx:ex,ey,sz:ez,old)
        else
           u(sx:ex,ey+1,sz:ez,old) =  rbuff_we(sx:ex,sz:ez,2)
        endif
     endif
!$OMP END SINGLE NOWAIT

     ! B-T send

!$OMP SINGLE
        if ( coords(3) > 0 ) then
           if ( dir > 0) then
              buff_bt(sx:ex,sy:ey,1) = u(sx:ex,sy:ey,sz,old)
           else
              u(sx:ex,sy:ey,sz-1,old) = rbuff_bt(sx:ex,sy:ey,1)
           endif
        endif
!$OMP END SINGLE NOWAIT

!$OMP SINGLE
     if( coords(3) < dims(3)-1) then
        if ( dir > 0 ) then
           buff_bt(sx:ex,sy:ey,2) = u(sx:ex,sy:ey,ez,old)  
        else
           u(sx:ex,sy:ey,ez+1,old) = rbuff_bt(sx:ex,sy:ey,2)
        endif
     endif
!$OMP  END SINGLE NOWAIT

   end subroutine buffs_halos_transfers


     subroutine exchange_halos
       implicit none

       integer status_sedrecv(MPI_STATUS_SIZE, 12), npoints, ierr
       ! send boundary data

        ! N-S send

!$OMP MASTER
        ! send to the right (i.e. s,e,t)
        if( coords(1) < dims(1)-1) then
           npoints = (ey - sy + 1 ) * ( ez - sz + 1)
           call MPI_ISend(buff_ns(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_s, &
                ns_tag, grid_comm, request(7), ierr)
        else
           request(7) = MPI_REQUEST_NULL
        endif

        ! send to the left
        if ( coords(1) > 0 ) then
           npoints = (ey - sy + 1 ) * ( ez - sz + 1)
           call MPI_ISend(buff_ns(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_n, &
                sn_tag, grid_comm, request(8), ierr)
        else
           request(8) = MPI_REQUEST_NULL
        endif

        ! W-E send

        ! send to the right (s,e,t)
        if( coords(2) < dims(2)-1) then
!           write(0,*) 'send to right ', myid, icol, npoints
            npoints = (ex - sx + 1 ) * ( ez - sz + 1)
           call MPI_ISend(buff_we(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_e, &
                we_tag, grid_comm, request(9), ierr)
        else
           request(9) = MPI_REQUEST_NULL
        endif

        ! send to the left
        if ( coords(2) > 0 ) then
!           write(0,*) 'send to left ', myid, icol, npoints
            npoints = (ex - sx + 1 ) * ( ez - sz + 1)
           call MPI_ISend(buff_we(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_w, &
                ew_tag, grid_comm, request(10), ierr)
        else
           ! with this values for unused request we can use mai_waitall at for communication finalisation
           request(10) = MPI_REQUEST_NULL
        endif

        ! B-T send

        ! send to the right (s,e,t)
        if( coords(3) < dims(3)-1) then
           npoints = (ex - sx + 1 ) * ( ey - sy + 1)
           call MPI_ISend(buff_bt(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_t, &
                bt_tag, grid_comm, request(11), ierr)
        else
           request(11) = MPI_REQUEST_NULL
        endif

        ! send to the left
        if ( coords(3) > 0 ) then
           npoints = (ex - sx + 1 ) * ( ey - sy + 1)
           call MPI_ISend(buff_bt(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_b, &
                tb_tag, grid_comm, request(12), ierr)
        else
           ! with this values for unused request we can use mai_waitall at for communication finalisation
           request(12) = MPI_REQUEST_NULL
        endif

        call MPI_Waitall(12, request, status_sedrecv,ierr)
        if ( ierr /= MPI_SUCCESS) then 
           write(0,*) 'error smoother waitall', coords, ierr
        endif

!$OMP END MASTER
       
     end subroutine exchange_halos

   end subroutine jacobi_smoother


   subroutine jacobi_smoother_cco( iteration, norm)
!$   use omp_lib
     use mpi
     implicit none
     
     integer, intent(in)   :: iteration
     real(wp), intent(out) :: norm

! Locals
     integer i, j, k, old, new, request(12)
     integer, save :: chnk
     real(wp) w, buff_ns(sy:ey, sz:ez, 2), &
                 rbuff_ns(sy:ey, sz:ez, 2), &
                 buff_we(sx:ex, sz:ez, 2), &
                 rbuff_we(sx:ex, sz:ez, 2), &
                 buff_bt(sx:ex, sy:ey, 2), &
                 rbuff_bt(sx:ex, sy:ey, 2)

     new = 3 - mod(iteration,2) - 1 ! start from on 
     old = 3 - new
     norm = 0.0_wp
     
      
      if ( iteration == 1) then 
         if (nthreads == 1) then 
            chnk = (ez - 1 - (sz + 1) + 1) * (ey - 1 - (sy + 1) + 1)
         else         
            chnk = ((ez - 1 - (sz + 1) + 1) * (ey - 1 - (sy + 1) + 1))/(nthreads-1)
         endif
      endif

!$OMP PARALLEL IF (nThreads > 1) DEFAULT(NONE) SHARED(request, buff_ns, rbuff_ns, &
!$OMP buff_we, rbuff_we, buff_bt, rbuff_bt, old, new, nthreads, iteration, &
!$OMP sx, ex, sy, ey, sz, ez, u, chnk) & 
!$OMP PRIVATE (i, j, k,  w) &
!$OMP REDUCTION (+:norm)


      !write(0,*) ' range  ', iteration, new, old, sr, er

     call post_recv

     call  buffs_halos_transfers(1, norm)
     
     call exchange_halos

     
     !$OMP DO SCHEDULE(DYNAMIC,chnk) COLLAPSE(2)
     do k = sz+1, ez-1
        do j = sy+1, ey-1
           do i = sx+1, ex-1
              w = inv6 * &
             (u(i-1,j,k,old) + u(i+1,j,k,old) +&
             u(i,j-1,k,old) + u(i,j+1,k,old) + &
             u(i,j,k-1,old) + u(i,j,k+1,old))
              norm = norm + w*w
        !w = exp(w/(1.0_wp+w*w))
              u(i,j,k,new) = w
           enddo
        enddo
     enddo
     !$OMP ENDDO NOWAIT

     call buffs_halos_transfers(-1, norm)
   
!$OMP END PARALLEL

       !write(0,*) 'did halos transfer', myrank, iteration, norm

   contains 

     subroutine post_recv
       implicit none

       integer npoints, ierr

!$OMP MASTER
       ! N-S
       ! receive ghost points for left face
       if ( coords(1) > 0 ) then
          npoints = (ey - sy + 1 ) * ( ez - sz + 1)
          call MPI_Irecv(rbuff_ns(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_n, &
               ns_tag, grid_comm, request(1), ierr)
       else
          request(1) = MPI_REQUEST_NULL
       endif

        ! receive ghost points for right face
       if (coords(1) < dims(1)-1) then
          !  receiving from right
          npoints = (ey - sy + 1 ) * ( ez - sz + 1)
          call MPI_Irecv(rbuff_ns(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_s, &
               sn_tag, grid_comm, request(2), ierr)
       else
           ! with this values for unused request we can use mai_waitall at for communication finalisation
          request(2) = MPI_REQUEST_NULL
       endif

        ! W-E recv

       if ( coords(2) > 0 ) then
          ! Post receives for the next color on the left face
          npoints = (ex - sx + 1 ) * ( ez - sz + 1)
          !           write(0,*) 'recv on left ' , myid,icol, npoints
          call MPI_Irecv(rbuff_we(:,:,1),npoints, MPI_DOUBLE_PRECISION, ngb_w,&
               we_tag, grid_comm, request(3), ierr)
       else
          request(3) = MPI_REQUEST_NULL
       endif
       ! receive ghost points for left face
       if (coords(2) < dims(2)-1) then
          !  receiving from right
          npoints = (ex - sx + 1 ) * ( ez - sz + 1)
          !            write(0,*) 'recv on right ' , myid,icol, npoints
          call MPI_Irecv(rbuff_we(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_e,&
               ew_tag, grid_comm, request(4), ierr)
       else
          ! with this values for unused request we can use mai_waitall at for communication finalisation
          request(4) = MPI_REQUEST_NULL
       endif
       
       ! B-T recv
       
       if ( coords(3) > 0 ) then
          ! Post receives for the next color on the left face
          npoints =  (ex - sx + 1 ) * ( ey - sy + 1)
          call MPI_Irecv(rbuff_bt(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_b,&
               bt_tag, grid_comm, request(5), ierr)
       else
          request(5) = MPI_REQUEST_NULL
       endif
       ! receive ghost points for left face
       if (coords(3) < dims(3)-1) then
          !  receiving from right
          npoints =  (ex - sx + 1 ) * ( ey - sy + 1)
          call MPI_Irecv(rbuff_bt(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_t,&
               tb_tag, grid_comm, request(6), ierr)
       else
          ! with this values for unused request we can use mai_waitall at for communication finalisation
          request(6) = MPI_REQUEST_NULL
       endif
       
!$OMP END MASTER
     end subroutine post_recv


     subroutine buffs_halos_transfers(dir, norm)
       implicit none
       integer, intent(in) :: dir !  1 grid -> buff
                                  ! -1 buff -> halo
       real(wp), intent(inout) :: norm

       integer i,j,k
       real(wp) w
       
       ! fill the transfer buffers, for black the updateis done in update face
! N-S

!$OMP MASTER
       if ( coords(1) > 0 ) then
          if ( dir > 0 ) then 
             buff_ns(sy:ey,sz:ez,1) = u(sx,sy:ey,sz:ez,old)
          else
             u(sx-1,sy:ey,sz:ez,old) = rbuff_ns(sy:ey,sz:ez,1)
          endif
       endif


     if( coords(1) < dims(1)-1) then
        if ( dir > 0) then 
           buff_ns(sy:ey,sz:ez,2) = u(ex,sy:ey,sz:ez,old)
        else
           u(ex+1,sy:ey,sz:ez,old) = rbuff_ns(sy:ey,sz:ez,2)
        endif
     endif    

! W-E

     if( coords(2) > 0) then
        if ( dir > 0 ) then 
           buff_we(sx:ex,sz:ez,1) = u(sx:ex,sy,sz:ez,old)
        else
          u(sx:ex,sy-1,sz:ez,old) =rbuff_we(sx:ex,sz:ez,1)
       endif
    endif

     if( coords(2) < dims(2)-1) then
        if ( dir > 0 ) then 
           buff_we(sx:ex,sz:ez,2) = u(sx:ex,ey,sz:ez,old)
        else
           u(sx:ex,ey+1,sz:ez,old) =  rbuff_we(sx:ex,sz:ez,2)
        endif
     endif

     ! B-T send


     if ( coords(3) > 0 ) then
        if ( dir > 0) then
           buff_bt(sx:ex,sy:ey,1) = u(sx:ex,sy:ey,sz,old)
        else
           u(sx:ex,sy:ey,sz-1,old) = rbuff_bt(sx:ex,sy:ey,1)
        endif
     endif


     if( coords(3) < dims(3)-1) then
        if ( dir > 0 ) then
           buff_bt(sx:ex,sy:ey,2) = u(sx:ex,sy:ey,ez,old)  
        else
           u(sx:ex,sy:ey,ez+1,old) = rbuff_bt(sx:ex,sy:ey,2)
        endif
     endif

! when data is received the outer shell of local  must be updated

     if (dir < 0) then 

        i = sx
        do k = sz+1, ez-1
           do j = sy+1, ey-1
              w = inv6 * &
                   (u(i-1,j,k,old) + u(i+1,j,k,old) +&
                   u(i,j-1,k,old) + u(i,j+1,k,old) + &
                   u(i,j,k-1,old) + u(i,j,k+1,old))
              norm = norm + w*w
              u(i,j,k,new) = w
           enddo
        enddo

        i = ex
        do k = sz+1, ez-1
           do j = sy+1, ey-1
              w = inv6 * &
                   (u(i-1,j,k,old) + u(i+1,j,k,old) +&
                   u(i,j-1,k,old) + u(i,j+1,k,old) + &
                   u(i,j,k-1,old) + u(i,j,k+1,old))
              norm = norm + w*w
              u(i,j,k,new) = w
           enddo
        enddo

       j = sy
       do k = sz+1, ez-1
          do i = sx, ex
             w = inv6 * &
                  (u(i-1,j,k,old) + u(i+1,j,k,old) +&
                  u(i,j-1,k,old) + u(i,j+1,k,old) + &
                  u(i,j,k-1,old) + u(i,j,k+1,old))
             norm = norm + w*w
             u(i,j,k,new) = w
          enddo
       enddo

        j = ey
        do k = sz+1, ez-1
           do i = sx, ex
              w = inv6 * &
                   (u(i-1,j,k,old) + u(i+1,j,k,old) +&
                   u(i,j-1,k,old) + u(i,j+1,k,old) + &
                   u(i,j,k-1,old) + u(i,j,k+1,old))
              norm = norm + w*w
              u(i,j,k,new) = w
           enddo
        enddo


        k = sz
        do j = sy, ey
           do i = sx, ex
              w = inv6 * &
                   (u(i-1,j,k,old) + u(i+1,j,k,old) +&
                   u(i,j-1,k,old) + u(i,j+1,k,old) + &
                   u(i,j,k-1,old) + u(i,j,k+1,old))
              norm = norm + w*w
              u(i,j,k,new) = w
           enddo
        enddo



        k = ez
        do j = sy, ey
           do i = sx, ex
              w = inv6 * &
                   (u(i-1,j,k,old) + u(i+1,j,k,old) +&
                   u(i,j-1,k,old) + u(i,j+1,k,old) + &
                   u(i,j,k-1,old) + u(i,j,k+1,old))
              norm = norm + w*w
              u(i,j,k,new) = w
           enddo
        enddo

     endif

!$OMP END MASTER
   end subroutine buffs_halos_transfers


     subroutine exchange_halos
       implicit none

       integer status_sedrecv(MPI_STATUS_SIZE, 12), npoints, ierr
       ! send boundary data

        ! N-S send

!$OMP MASTER
        ! send to the right (i.e. s,e,t)
        if( coords(1) < dims(1)-1) then
           npoints = (ey - sy + 1 ) * ( ez - sz + 1)
           call MPI_ISend(buff_ns(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_s, &
                ns_tag, grid_comm, request(7), ierr)
        else
           request(7) = MPI_REQUEST_NULL
        endif

        ! send to the left
        if ( coords(1) > 0 ) then
           npoints = (ey - sy + 1 ) * ( ez - sz + 1)
           call MPI_ISend(buff_ns(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_n, &
                sn_tag, grid_comm, request(8), ierr)
        else
           request(8) = MPI_REQUEST_NULL
        endif

        ! W-E send

        ! send to the right (s,e,t)
        if( coords(2) < dims(2)-1) then
!           write(0,*) 'send to right ', myid, icol, npoints
            npoints = (ex - sx + 1 ) * ( ez - sz + 1)
           call MPI_ISend(buff_we(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_e, &
                we_tag, grid_comm, request(9), ierr)
        else
           request(9) = MPI_REQUEST_NULL
        endif

        ! send to the left
        if ( coords(2) > 0 ) then
!           write(0,*) 'send to left ', myid, icol, npoints
            npoints = (ex - sx + 1 ) * ( ez - sz + 1)
           call MPI_ISend(buff_we(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_w, &
                ew_tag, grid_comm, request(10), ierr)
        else
           ! with this values for unused request we can use mai_waitall at for communication finalisation
           request(10) = MPI_REQUEST_NULL
        endif

        ! B-T send

        ! send to the right (s,e,t)
        if( coords(3) < dims(3)-1) then
           npoints = (ex - sx + 1 ) * ( ey - sy + 1)
           call MPI_ISend(buff_bt(:,:,2), npoints, MPI_DOUBLE_PRECISION, ngb_t, &
                bt_tag, grid_comm, request(11), ierr)
        else
           request(11) = MPI_REQUEST_NULL
        endif

        ! send to the left
        if ( coords(3) > 0 ) then
           npoints = (ex - sx + 1 ) * ( ey - sy + 1)
           call MPI_ISend(buff_bt(:,:,1), npoints, MPI_DOUBLE_PRECISION, ngb_b, &
                tb_tag, grid_comm, request(12), ierr)
        else
           ! with this values for unused request we can use mai_waitall at for communication finalisation
           request(12) = MPI_REQUEST_NULL
        endif

        call MPI_Waitall(12, request, status_sedrecv,ierr)
        if ( ierr /= MPI_SUCCESS) then 
           write(0,*) 'error smoother waitall', coords, ierr
        endif

!$OMP END MASTER
       
     end subroutine exchange_halos

   end subroutine jacobi_smoother_cco


   subroutine initContext
     implicit none
    
    integer i, j, argc, ierr
    character(len=128) buff

    !"Logicals" for file output
    vOut = .false.
    test_computation = .false.; pHeader = .true.; 
    pContext = .false.
    
    ! Default required values
    ngxyz(:) = (/ 33, 33, 33 /); niter = 20;  nthreads = 1; nproc = 1;
    npxyz(:) = (/ 1, 1, 1 /)
    
    use_cco = .false.

    ! get the number of command arguments
    argc = command_argument_count()

!    write(0,*) 'argc ', argc

    i = 0
    do 
       i = i + 1

       if ( i > argc ) exit

       call get_command_argument(i,value=buff)

!       write(0,*) 'buff ', trim(buff)
       
       if (trim(buff) == "-ng") then
          do  j = 1, 3
             i = i + 1
             call get_command_argument(i,value=buff)
             read(buff,*) ngxyz(j)
          enddo
      else if (trim(buff) == "-np") then
         do j = 1, 3
            i = i + 1
            call get_command_argument(i,value=buff)
            read(buff,*) npxyz(j)
         enddo
      else if (trim(buff) == "-niter") then
         i = i + 1
         call get_command_argument(i,value=buff)
         read(buff,*) niter
      else if (trim(buff) == "-v") then
         ! Look for "verbose" standard out
         vOut = .true.
      else if (trim(buff) == "-nh") then
         ! Look for "No Header" option
         pHeader = .false.
      else if (trim(buff) == "-pc") then
         ! Look for "Print Context" option
         pContext = .true.
      else if (trim(buff) == "-use_cco") then
        ! Use computation communication overlap
         use_cco = .true.
        else if(trim(buff) == "-t" ) then 
           test_computation = .true.
      else if (trim(buff) == "-help" .or. trim(buff) == "--help") then
         call get_command_argument(0,value=buff)
         write(*,'(a,a,a,3(a))') & 
              "Usage: ",  trim(buff), " [-ng grid-size-x grid-size-y grid-size-z]", &
               "[-np num-proc-x num-proc-y num-proc-z]", &
               "[-niter num-iterations] ", &
               "[-v] [-t] [-pc] [-use_cco] [-nh] [-help]"
         call mpi_finalize(ierr)
         stop
      else
         write(*,'(a,a,a)') &
              " Wrong option ", trim(buff), " try -help"
         call mpi_finalize(ierr)
         stop
      endif
         
   enddo

 end subroutine initContext
 
 
 subroutine setPEsParams 
     use mpi
!$     use omp_lib
     implicit none

     integer ierr

     ! Find nPEs 
     
     call MPI_Comm_size(MPI_COMM_WORLD, nproc,ierr)
     ! Find myPE
     call MPI_Comm_rank(MPI_COMM_WORLD, myrank,ierr); 

     if ( npxyz(1) * npxyz(2) * npxyz(3) /= nproc ) then 
        write(0,*) "wrong number of processors in mpiexe argument"
        call mpi_abort(mpi_comm_world, 1, ierr)
     endif

     ! generate 3d topology

     call mpi_cart_create(mpi_comm_world, 3, npxyz , &
          (/ .false., .false., .false. /), .false., grid_comm, ierr)
     
     ! split the grid to ranks
     call compute_local_grid_ranges

     ! Find number of threads per task 

!$     nThreads = omp_get_max_threads()

   contains

     subroutine compute_local_grid_ranges
       implicit none
       
       integer i, nl(3), local_shift(3), ux(3), uy(3), uz(3), r
       logical periods(3)

       call mpi_cart_get(grid_comm, 3, dims, periods, coords, ierr)
       
       nl(:) = ngxyz(:)/dims(:)

       
      ! test for silly values ?

      do i=1,3

         r = mod(ngxyz(i),dims(i))

         if ( coords(i) < r ) then
            nl(i) = nl(i)+1
            local_shift(i) = coords(i)*nl(i)
         else
            local_shift(i) = r * (nl(i)+1) + (coords(i) - r) * nl(i)
         endif

      end do

      sx = local_shift(1) + 1
      ex = sx + nl(1) - 1
      sy = local_shift(2) + 1
      ey = sy + nl(2) -1
      sz = local_shift(3) + 1
      ez = sz + nl(3) -1

      if ( ex - sx + 1 < 3 .or. &
           ey - sy + 1 < 3 .or. &
           ez - sz + 1 < 3 ) then
         write(0,*) " local domain too small, please choose a different MPI topology or grid sizes"
         write(0,*) "rank=", myrank, ", coords=", coords,&
              ", local grid start-end", sx, ex ,sy, ey, sz, ez
         call mpi_abort(mpi_comm_world, 1,ierr)
      endif
      
      ! sx, ex, ... are internal points
      ! therefore we need to  shift them if the rank
      ! has sides on domain boundary
      if ( coords(1) == 0          ) sx = sx + 1
      if ( coords(1) == dims(1) -1 ) ex = ex - 1
      if ( coords(2) == 0          ) sy = sy + 1
      if ( coords(2) == dims(2) -1 ) ey = ey - 1
      if ( coords(3) == 0          ) sz = sz + 1
      if ( coords(3) == dims(3) -1 ) ez = ez - 1
 
      !write(0,*) 'grid ', coords, sx,ex,sy,ey,sz,ez

      ! get  nearest neighbors ranks
      ux = (/ 1, 0 , 0 /)
      uy = (/ 0, 1, 0 /)
      uz = (/ 0, 0, 1 /)

      if (coords(1) > 0) then 
         call mpi_cart_rank(grid_comm, coords - ux, ngb_n, ierr)
      else
         ngb_n = MPI_PROC_NULL
      endif

      if (coords(1) < dims(1) -1) then 
         call mpi_cart_rank(grid_comm, coords + ux, ngb_s, ierr)
      else
         ngb_s = MPI_PROC_NULL
      endif

       if (coords(2) > 0) then 
         call mpi_cart_rank(grid_comm, coords - uy, ngb_w, ierr)
      else
         ngb_w = MPI_PROC_NULL
      endif

      if (coords(2) < dims(2) -1) then 
         call mpi_cart_rank(grid_comm, coords + uy, ngb_e, ierr)
      else
         ngb_e = MPI_PROC_NULL
      endif

       if (coords(3) > 0) then 
         call mpi_cart_rank(grid_comm, coords - uz, ngb_b, ierr)
      else
         ngb_b = MPI_PROC_NULL
      endif

      if (coords(3) < dims(3) -1) then 
         call mpi_cart_rank(grid_comm, coords + uz, ngb_t, ierr)
      else
         ngb_t = MPI_PROC_NULL
      endif

     end subroutine compute_local_grid_ranges


   end subroutine setPEsParams


   subroutine initial_field
     implicit none
     
     integer i, j, k

     allocate ( u(sx-1:ex+1, sy-1:ey+1, sz-1:ez+1, 2) )
   
     ! first touch
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k) SCHEDULE(STATIC) 
     do k = sz-1, ez+1
        do j = sy-1, ey+1
           do i = sx-1, ex+1
              if ( ( i == sx - 1 .or. i == ex + 1 ) .or. &
                   ( j == sy - 1 .or. j == ey + 1 ) .or. &
                   ( k == sz - 1 .or. k == ez + 1 ) ) then 
                 u(i, j, k, 2) = 0.0_wp
              else
                 u(i, j, k, 1) = 0.0_wp
                 u(i, j, k, 2) = sin(pi*kx*(i-1)/real(ngxyz(1)-1, wp)) * &
                                 sin(pi*ky*(j-1)/real(ngxyz(2)-1, wp)) * &
                                 sin(pi*kz*(k-1)/real(ngxyz(3)-1, wp))
              endif
           enddo
        enddo
     enddo
!$OMP END PARALLEL DO
     

 

   end subroutine initial_field


   subroutine printContext 
     implicit none
     print "(a,3I5)","Global grid sizes :   ", ngxyz
     print "(a,3I5)","MPI    topology   :   ", npxyz 
     print "(a,I5)" ,"Number of Iterations: ", NITER

     if (pHeader) then
        print "(a)", "Summary Standard Ouput with Header "
     else
        print "(a)", "Summary Standard Output without Header "
     endif

     if (vOut) then 
        print "(a)", "Verbose Output "
     endif

     if (use_cco) then
        print "(a)", "Using computation-communication overlap "
     endif

   end subroutine printContext


   subroutine check_norm(iter, norm)
     use mpi
     implicit none
     integer, intent(in)  :: iter
     real(wp), intent(in) :: norm

     integer ierr
     real(kind(0.d0)) ln, gn, r, eig
     real(kind(0.d0)), save :: prev_gn

     if ( iter == 1 ) then
        prev_gn =1.0d0
        if ( myrank == ROOT) then 
           print "(a)", "correctness check"
           print "(a)", "iteration, norm ratio, deviation from eigenvalue"
        endif
     endif

     ln = norm
     call mpi_reduce(ln, gn, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
          root, mpi_comm_world, ierr)
     
     if ( myrank == root ) then
        r = sqrt(gn/prev_gn)
        eig = (cos(pi*kx/(ngxyz(1)-1)) + cos(pi*ky/(ngxyz(2)-1)) + cos(pi*kz/(ngxyz(3)-1)))/3.0_wp
        if ( iter > 1) then
           print "(I5,4x,E12.5,5x,E12.5)" , iter, r, r - eig
        endif
        
        prev_gn = gn
     endif

   end subroutine check_norm


   subroutine timeUpdate( times)
     use mpi
     implicit none
     real(kind(0.d0)) times(:,:)

     integer ierr

     if (myrank == ROOT ) then
       call MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, times, NITER, MPI_DOUBLE_PRECISION, ROOT, MPI_COMM_WORLD, ierr)
      else
       call MPI_Gather(times, NITER, MPI_DOUBLE_PRECISION, MPI_BOTTOM, 0, MPI_DATATYPE_NULL, ROOT, MPI_COMM_WORLD, ierr)
      endif

   end subroutine timeUpdate


   subroutine statistics( times, minTime, meanTime, maxTime, stdvTime, NstdvTime)
     implicit none
     real(kind(0.d0)), intent(in) :: times(:,:)
     real(kind(0.d0)),intent(out) :: minTime, meanTime, maxTime, &
      stdvTime, NstdvTime

     integer shift

     if ( niter > 10 )  then
        !eliminate startup baias if niter is large enough
        shift = 5 
     else
        shift = 0
     endif
     
!Compute mean, max, min of times and standard deviation

     meanTime  = sum(times(1+shift:,:))/(real(nproc,wp) * real(NITER-shift))
     minTime = minval(times(1+shift:,:))
     maxTime  = maxval(times(1+shift:,:))
     
     stdvTime = sum((times(1+shift:,:) - meanTime)**2)
     stdvTime = sqrt(stdvTime/(real(nproc) * real(NITER-shift) - 1.0))

     NstdvTime = stdvTime / meanTime

   end subroutine statistics


   subroutine stdoutIO(times, minTime, meanTime, maxTime, NstdvTime, norm)
     implicit none
     real(kind(0.d0)), intent(in) :: times(:,:)
     real(kind(0.d0)),intent(in) :: minTime, meanTime, maxTime,  NstdvTime
     real(wp), intent(in) :: norm

     integer funit
     character(len=128) fmtstr

! unit number to write
     funit = 6
     
     if ( pHeader ) then
        write(funit,'(/,a,g22.15,/)') "Last norm ", sqrt(norm)
! Print heading and summary data
       write(funit,'(a)') "#==================================================&
            &=============================================================================#"
       write(funit,'(a2,8(a9,1x),4(a10,1x),a)') "# ", "NPx", "NPy", "NPz" , &
            &"Threads", "Nx", "Ny", "Nz","NITER","meanTime", "maxTime","minTime","NstdvTime","  #"
       write(funit,'(a)') "#=================================================&
            &==============================================================================#"
    endif
    write(funit,'(2x,8(I9,1x),4(1x,E9.3,1x))')  npxyz, NThreads, ngxyz , &
         NITER, meanTime, maxTime, minTime, NstdvTime
     

! Only if "Verbose Output" asked for 
     if ( vOut ) then 
       write(funit,*) "# Full Time Output (rows are times, cols are tasks) "
       write(fmtstr,'(a,I0,a,a)') '(',nproc,'(E10.4,x)',')'
     ! Print full times matrix

       write(funit, fmtstr) transpose(times(:,:))
      end if

    end subroutine stdoutIO
 
! not used 
!!$  subroutine fileIO(times, covar, minTime, meanTime, maxTime, NstdvTime)
!!$     implicit none
!!$     real(kind(0.d0)), intent(in) :: times(:,:)
!!$     real(kind(0.d0)),intent(in) :: minTime, meanTime, maxTime,  NstdvTime, covar(:,:)
!!$
!!$     integer funit
!!$     character(len=128) fmtstr
!!$
!!$! unit number
!!$        funit = 33
!!$! format string
!!$     write(fmtstr,'(a,I0,a,a)') '(',nproc,'(E15.9,x)',')'
!!$      
!!$      if (fOut) then
!!$       open(funit,file=fileOutName,status="replace")
!!$! Print heading and summary data
!!$       write(funit,'(a)') "#===================================================================================================#"
!!$       write(funit,'(a2,10a10,a)') "# ", "Tasks", "MPI topology", "Threads",&
!!$            & "Grid","NITER","meanTime", "maxTime","minTime","NstdvTime","  #"
!!$       write(funit,'(a)') "#===================================================================================================#"
!!$       write(funit,'(2x,9(I8,2x),4(E9.3,1x))') nproc, npxyz, NThreads, ngxyz , NITER, meanTime, maxTime, minTime, NstdvTime
!!$       write(funit,'(/)')
!!$       write(funit,*) "# Full Time Output (rows are times, cols are tasks) "
!!$! Print full times matrix
!!$       write(funit, fmtstr) transpose(times(:,:))
!!$       close(funit)
!!$      endif
!!$
!!$      if (tOut) then
!!$        open(unit=funit, file=timesOutName, status="replace")
!!$        write(funit,*) "# Full Time Output (rows are times, cols are tasks) "
!!$        write(funit, fmtstr) transpose(times(:,:))
!!$        close(funit)
!!$      endif
!!$
!!$   end subroutine fileIO

   
 end module functions


