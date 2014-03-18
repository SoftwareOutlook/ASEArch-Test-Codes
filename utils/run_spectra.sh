#!/bin/sh
#
# collects data accross grid size for specified algorithms of Jacobi Test Code 
# 
# Lucian Anton 
# March 2014
#

# command options
options=$@

# read options
for argument in $options
  do
    index=$((index + 1))

    # check if we have a flag
    flag="${argument%=*}"
    val="${argument#*=}"
    case $flag in
        -model) model_list="${val//,/ }" ;;
        -wave) wave_params=(${val//,/ }) ;;
	-blocks) blk_val=(${val//,/ }) ; blk_x="${blk_val[0]}";blk_y="${blk_val[1]}";blk_z="${blk_val[2]}";;
	-maxsize)max_linsize=$val;;
	-step)step=$val;;
        -exe)exe_list="${val//,/ }";;
	-nthreads)threads_list="${val//,/ }" ;;
	-test)test_flag="-t";;
        -malign)fmalign="-malign $val";;
	-system)run_command=$val ;;
	-help|-h) 
	       echo "$0 -model=<model list> wave=<wavel param list> -blocks=<block sizes list>"
	       echo "   -mingrid=<start grid sizes> -maxgrid=<end grid sizes> -step=<grid increase>"
	       echo "   -exe=<executables list> -nthreads=<number of OpenMP threads list> "
	       echo "   -test pass the -t flag to executable for testing"
	       echo "   -malig=<val> use posix_memalign for main arrays, align memory with <va>"
	       echo "   -system=<val> : select system <val> for bach execution" 
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

[ -z "$model_list" ] && model_list="baseline-opt blocked wave"
if [ -z "$blk_val" ]
then 
    blk_x=0; blk_y=0; blk_z=0
fi

[ -z "$max_linsize" ] && max_linsize=$MAX_LINSIZE

[ -z "$step" ] && step=7

[ -z "$threads_list" ] && threads_list=1

#echo "max linsize $max_linsize step $step model_list $model_list"

index=0
for exe in $exe_list
do
    for model in $model_list
    do
	for nth in $threads_list 
	do
	    export OMP_NUM_THREADS=$nth
	    #fout=out_"$model"_"${exe%.exe}"_t"$nth".txt
	    #fspectra=${fout%.txt}_spectra.txt 
	
	    #[ -f "$fout" ] && cat /dev/null > $fout

	    echo "# $((index++)) model $model nth $nth exe $exe" >> $fout
	    # $((++index))
	    for ((linsize=10; linsize <= max_linsize; linsize += step)) 
	    do
		niter=$(((10*max_linsize)/linsize))
		nruns=5
		if [ "$model" = wave ] 
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
		    
		
		if [ "$blk_x" -eq 0 ] ; then  blk_xt=$linsize ; else blk_xt=$blk_x ; fi
		if [ "$blk_y" -eq 0 ] ; then  blk_yt=$linsize ; else blk_yt=$blk_y ; fi
		if [ "$blk_z" -eq 0 ] ; then  blk_zt=$linsize ; else blk_zt=$blk_z ; fi
		
		arguments="-ng $linsize $linsize $linsize -nb $blk_xt $blk_yt $blk_zt -model $model $wave_params_temp -niter $niter -nruns $nruns -nh $test_flag  $fmalign"

		echo "nth $nth $exe $arguments"
		case $run_command in
		    mic)
		    # mic on csemic2
			export I_MPI_MIC=1
			mpirun -n 1 -host mic0  -env LD_LIBRARY_PATH /lib -env OMP_NUM_THREADS $nth -env KMP_AFFINITY verbose,balanced "$exe" $arguments  >> $fout
			;;
		    bgq)
		    # blue joule
			/bgsys/drivers/ppcfloor/hlcs/bin/runjob -n 1 --envs OMP_NUM_THREADS="$nth" BG_THREADLAYOUT=1 : "$exe" $arguments  >> $fout
			;;
                    idp)
		    # IdataPlex
			mpiexec.hydra -np 1 -env OMP_NUM_THREADS $nth  -env KMP_AFFINITY verbose,granularity=core,scatter "$exe" $arguments  >> $fout
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
