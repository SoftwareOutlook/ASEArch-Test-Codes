!
! DL_HOMB fortran version, see functions_c.c and homb_c.c for more details
!
! Lucian Anton 8/07/2013


!!$
!!$
!!$  Copyright 2009 Maxwell Lipford Hutchinson
!!$
!!$  This file is part of HOMB.
!!$
!!$  PGAF is free software: you can redistribute it and/or modify
!!$  it under the terms of the GNU General Public License as published by
!!$  the Free Software Foundation, either version 3 of the License, or
!!$  (at your option) any later version.
!!$
!!$  PGAF is distributed in the hope that it will be useful,
!!$  but WITHOUT ANY WARRANTY; without even the implied warranty of
!!$  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!$  GNU General Public License for more details.
!!$
!!$  You should have received a copy of the GNU General Public License
!!$  along with PGAF.  If not, see <http://www.gnu.org/licenses/>.
!!$ 
!!$


program main 
  use mpi
  use functions
  implicit none
  ! State information 
  integer  iter, ierr, istat

  ! MPI thread safety level parameters
  integer, parameter :: requested_mpi_safety = MPI_THREAD_FUNNELED
  integer provided_mpi_safety

  ! L2 norms 
  real(wp) norm, gnorm

  ! Timing measurements 
  real(kind(0.d0)) :: startTime, endTime
  real(kind(0.d0)) :: meanTime = 0.0d0, minTime = huge(0.d0), &
    maxTime = 0.d0 , stdvTime = 0.0d0, NstdvTime = 0.0d0
  real(kind(0.d0)),allocatable :: times(:,:)


  ! Initialize MPI 
   call MPI_Init_thread(requested_mpi_safety, provided_mpi_safety, ierr)

   if ( requested_mpi_safety /= provided_mpi_safety ) then
     write(*,'(a,/a,I0,/,a,I0)') " Warning, MPI thread safety requested &
          & level is not equal with provided ",  &
          & " requested ", requested_mpi_safety, &
          & " provided  ", provided_mpi_safety
  endif

  ! Initialize Global Context 
   call initContext

  ! Get task/thread information 
  call setPEsParams
  
  ! Create matrix for times 
  if (myrank == ROOT) then
     allocate(times(NITER,0:nproc-1), stat=istat)
  else 
     allocate(times(NITER,myrank:myrank), stat=istat)
  endif
  if ( istat /= 0 ) then 
     print*, " error in allocation of times matrix, quitting ...!"
     call MPI_Abort(MPI_COMM_WORLD,istat,ierr)
  else
    times = 0.d0
  endif

  ! Initialize grid 
  call initial_field

  !  Print Global Context to standard out 
  if (myrank == ROOT .and. pContext) then   
     call printContext
  endif

  ! Solve 
  do iter = 1, NITER
     
! Begin Timing 
   startTime = MPI_Wtime()
     
    ! Do all the real work
    if (use_cco) then
       call jacobi_smoother_cco(iter, norm)
    else
      call jacobi_smoother(iter, norm)
     endif
    
    !End timing 
    endTime = MPI_Wtime()
    
! Store Time 
    times(iter,myrank) = endTime-startTime

    if ( test_computation) then
       call check_norm(iter, norm)
    end if

 end do

  ! Gather iteration runtimes to ROOT's matrix 
  call timeUpdate(times)

  ! Run statistics on times (Root only) 
  if (myrank == ROOT) then
     call statistics(times, minTime, meanTime, &
          maxTime, stdvTime, NstdvTime)
  endif

  !compute the final global norm, useful for quick validation
  call mpi_reduce(norm, gnorm, 1, MPI_DOUBLE_PRECISION,&
       MPI_SUM, ROOT, MPI_COMM_WORLD, ierr)

  ! Output 
  if (myrank == ROOT) then
        call stdoutIO(times, minTime, meanTime, maxTime, NstdvTime, gnorm)
  endif

  ! Finalize 
  call MPI_Finalize(ierr);

end program main

