#!/bin/sh
#
# collects data accross grid size for specified algorithms of Jacobi Test Code 
# 
# Lucian Anton 
# March 2014
#

# read options
for argument in "$@"
  do
    index=$((index + 1))

    # check if we have a flag
    flag="${argument%%=*}"
    val="${argument#*=}"
    case $flag in
        -cmodel) cmod_val=$val ;;
        -alg) alg_list="${val//,/ }" ;;
        -wave) wave_params=(${val//,/ }) ;;
	-blocks) blk_val=(${val//,/ }) ; blk_x="${blk_val[0]}";blk_y="${blk_val[1]}";blk_z="${blk_val[2]}";;
	-maxsize)max_linsize=$val;;
	-minsize)min_linsize=$val;;
	-step)step=$val;;
        -exe)exe_list="${val//,/ }";;
	-nthreads)threads_list="${val//,/ }" ;;
	-nproc)proc_list="${val//,/ }" ;;
	-test)test_flag="-t";;
        -malign)fmalign="-malign $val";;
	-system)run_command=$val ;;
        -run-opts)run_opts="$val" ;;
        -info)print_context=1 ;;
	-help|-h) 
	       echo ""
	       echo "$0 -cmodel=<compute model> -alg=<algorithms list> wave=<wavel param list> -blocks=<block sizes list>"
	       echo "   -minsize=<start grid sizes> -maxsize=<end grid sizes> -step=<linear grid size increase>"
	       echo "   -exe=<executables list> -nthreads=<number of OpenMP threads list>, -nproc=<number of MPI ranks in each dimension>"
	       echo "   -test pass the -t flag to executable for testing"
	       echo "   -malig=<val> use posix_memalign for main arrays, align memory with <val>"
	       echo "   -system=<val> : select system <val> for batch execution" 
	       exit 0
	       ;;
        *)  echo "unknown flag $argument" ; exit  1;;
    esac
done

#defaults
# default max grid
MAX_LINSIZE=32
MIN_LINSIZE=32
#data file
fout=jacobi_spectra_$(date "+%s").txt

if [ -z "$exe_list" ] 
then 
    echo "please provide an executable list:  -exe=exe1[,exe2[,..]]" 
    exit 1
fi

[ -z "$cmod_val"    ] && cmod_val=openmp
[ -z "$alg_list"    ] && alg_list=baseline
[ -z "$min_linsize" ] && min_linsize=$MIN_LINSIZE
[ -z "$max_linsize" ] && max_linsize=$MAX_LINSIZE

[ -z "$step" ] && step=7

[ -z "$threads_list" ] && threads_list=1
[ -z "$proc_list" ]    && proc_list="1 1 1"
aux=($proc_list)
nproc=$((aux[0] * aux[1] * aux[2]))


#echo "min-max linsize $min_linsize $max_linsize step $step model_list $model_list exe list $exelist"

index=0
for exe in $exe_list
do
    # this might fail if login nodes cannot run executable compiled for compute nodes 
    echo "# $exe version "$( $exe -version ) >> $fout
    for alg in $alg_list
    do
	for nth in $threads_list 
	do
	    export OMP_NUM_THREADS=$nth

	    echo "# $((index++)) compute model $cmod_val algorithm $alg nth $nth MPI ranks $proc_list exe $exe" >> $fout
	    # $((++index))
	    for ((linsize=min_linsize; linsize <= max_linsize; linsize += step)) 
	    do
                nitermax=10
                nitersize=$(((10*max_linsize)/linsize))
		niter=$nitersize #$((nitersize>nitermax?nitermax:nitersize))
		nruns=5
		if [ "$alg" = wave ] 
		then
		    if [ -z "$wave_params" ] 
		    then 
			wave_params_temp="$niter $((nth > 1 ? 2 : 1))"
		    else
			# adjsut run parameters to avoid unnecessary stops
			(( wave_params[0] > niter )) && niter=${wave_params[0]}
			(( niter % wave_params[0] > 0 )) && niter=$(( niter -  niter % wave_params[0] ))
			wave_params_temp="$niter $(( wave_params[1] > nth ? nth : wave_params[1] ))"
		    fi
		fi
		  
		# print context info at the top of the file if -info flag is set
		if [ -n "$print_context" ]
		then
		    if [ -z "$first_time_inner_loop" ] ; then pc=-pc ; first_time_inner_loop=done ; else pc=""; fi
		fi
  
		arguments="-np $proc_list $pc -cmodel $cmod_val -ng $linsize $linsize $linsize -alg $alg $wave_params_temp -niter $niter -nruns $nruns -nh $test_flag  $fmalign"
		
		# block flags are not not compulsory
		if [ "$blk_val" ] 
		then
		    if [ "$blk_x" -eq 0 ] ; then  blk_xt=$linsize ; else blk_xt=$blk_x ; fi
		    if [ "$blk_y" -eq 0 ] ; then  blk_yt=$linsize ; else blk_yt=$blk_y ; fi
		    if [ "$blk_z" -eq 0 ] ; then  blk_zt=$linsize ; else blk_zt=$blk_z ; fi
		    arguments="$arguments -nb  $blk_xt $blk_yt $blk_zt"
		fi

		echo "nth $nth run opts $run_opts $exe $arguments"
		case $run_command in
		    mic)
		    # mic on csemic2
			export I_MPI_MIC=1
                        # -env KMP_AFFINITY balanced
			mpirun -n $nproc -host mic0 $run_opts -env LD_LIBRARY_PATH /lib -env OMP_NUM_THREADS $nth -env KMP_AFFINITY balanced "$exe" $arguments  >> $fout
			;;
		    bgq)
		    # blue joule
			/bgsys/drivers/ppcfloor/hlcs/bin/runjob -n $nproc $run_opts --envs OMP_NUM_THREADS="$nth" BG_THREADLAYOUT=1 : "$exe" $arguments  >> $fout
			;;
                    idp)
		    # IdataPlex
			mpiexec.hydra -np $nproc $run_opts -env OMP_NUM_THREADS $nth "$exe" $arguments  >> $fout
                        ;;
		    mpich)
			# interactive mpi
			mpiexec -n $nproc $run_opts $exe $arguments >> $fout
			;;
                    cray)
                    # Cray systems use aprun
                        aprun -n $nproc -d $OMP_NUM_THREADS $run_opts "$exe" $arguments  >> $fout
                        ;;
                    slurm)
                        srun -n $nproc -c $OMP_NUM_THREADS $run_opts "$exe" $arguments  >> $fout
                        ;;
		    *)
		    # interactive shell 
			$exe $arguments  >> $fout
			;;
		esac
	    done
	    # make a gnuplot block
            echo " " >> $fout
            echo " " >> $fout
	done
    done	
done	
