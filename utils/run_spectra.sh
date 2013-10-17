#!/bin/sh
#
# collect data accross grid size for a list of models 
#

# default max grid
MAX_LINSIZE=30
fout=jacobi_spectra_$(date "+%s").txt

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
	-blocks) blk_val=(${val//,/ }) ; blk_x="${blk_val[0]}";blk_y="${blk_val[1]}";blk_z="${blk_val[2]}";;
	-maxsize)max_linsize=$val;;
	-step)step=$val;;
        -exe)exe_list="${val//,/ }";;
	-nthreads)threads_list="${val//,/ }" ;;
	-test)test_flag="-t";;
        *)  echo "unknown flag $argument" ; exit  1;;
    esac
done

#defaults
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
		biter=$(((10*max_linsize)/linsize))
		niter=5
		if [ "$model" = wave ]
		then 
		    nwave="$biter $((nth<biter?nth:biter))"
		    echo "model $model $nwave"
		else
		    nwave=""
		fi
		
		if [ "$blk_x" -eq 0 ] ; then  blk_xt=$linsize ; else blk_xt=$blk_x ; fi
		if [ "$blk_y" -eq 0 ] ; then  blk_yt=$linsize ; else blk_yt=$blk_y ; fi
		if [ "$blk_z" -eq 0 ] ; then  blk_zt=$linsize ; else blk_zt=$blk_z ; fi
		
		echo "./"$exe" -ng $linsize $linsize $linsize -nb $blk_xt $blk_yt $blk_zt -model $model $nwave -biter $biter -niter $niter -nh $test_flag  >> $fout"
		./"$exe" -ng $linsize $linsize $linsize -nb $blk_xt $blk_yt $blk_zt -model $model $nwave -biter $biter -niter $niter -nh $test_flag  >> $fout
#/bgsys/drivers/ppcfloor/hlcs/bin/runjob -n $nproc -p $NTASK --envs OMP_NUM_THREADS="$nth" BG_THREADLAYOUT=1 : "$exe" -ng $linsize $linsize $linsize -nb $blk_xt $blk_yt $blk_zt -model $model $nwave -biter $biter -niter $niter -nh $test_flag  >> $fout -t
	    done
	    # make a gnuplot block
            echo " " >> $fout
            echo " " >> $fout
#	    awk '{print $2, (($2-1)*($3-1)*($4-1))/$'"$tmincol"'}' $fout > $fspectra
	done
    done	
done	
