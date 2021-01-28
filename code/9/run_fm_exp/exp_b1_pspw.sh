#!/bin/bash

(./build-binary.sh)

data_path=$1
mode=$2
pos_bias=$3

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
	echo "Plz input data_path & mode & pos_bias!!!!!"
	exit 0 
fi

root=`pwd`
run_exp(){
	cdir=$1
	rdir=$2
	mode=$3
	cmd="cd ${cdir}"
	cmd="${cmd}; ./grid.sh" 
	#cmd="${cmd}; ./do-test.sh ${mode}"
	#cmd="${cmd}; echo 'va_logloss va_auc' > ${mode}.record"
	#cmd="${cmd}; python select_params.py logs ${mode} | rev | cut -d' ' -f1-2 | rev >> ${mode}.record"
	#cmd="${cmd}; tail -n2 test-score.${mode}/*.log | rev | cut -d' ' -f1-2 | rev >> ${mode}.record"
	cmd="${cmd}; cd ${rdir}"
	echo ${cmd}
}

run_exp_imp(){
	cdir=$1
	rdir=$2
	mode=$3
	imp_type=$4
	imp_dir=$5
	cmd="cd ${cdir}"
	cmd="${cmd}; ./grid.sh ${imp_type} ${imp_dir} ${mode}" 
	cmd="${cmd}; ./do-test.sh ${imp_type} ${imp_dir} ${mode}"
	cmd="${cmd}; echo 'va_logloss va_auc' > ${mode}.record" 
	cmd="${cmd}; tail -n1 ${mode}.top | rev | awk -F' ' '{print \$1,\$2}' | rev >> ${mode}.record" 
	cmd="${cmd}; tail -n1 test-score.${mode}/*.log | rev | awk -F' ' '{print \$1,\$2}' | rev >> ${mode}.record" 
	cmd="${cmd}; cd ${root}"
	echo ${cmd}
}

set -e
exp_dir=`basename ${data_path}`

for i in '.'
do 
	for k in 0.01 0.1
	do 
		cdir=${exp_dir}/der${i}${k}.ps
		mkdir -p ${cdir}
		ln -sf ${root}/scripts/init.sh ${cdir}
		ln -sf ${root}/scripts/grid-ps.sh ${cdir}/grid.sh
		ln -sf ${root}/scripts/select_params_ps.py ${cdir}/select_params.py
		ln -sf ${root}/tfboys/tfboys-complex-imp-grid/train ${cdir}/hytrain
		ln -sf ${root}/${data_path}/der${i}${k}/item.ffm ${cdir}
		for j in 'trva' 'tr' 'va'
		do
			ln -sf ${root}/${data_path}/der${i}${k}/select_${j}.ffm ${cdir}/${j}.ffm
		done
		run_exp ${cdir} ${root} ${mode} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
	done
done

for imp_type in 'pspw'
do
	for i in '.comb.' 
	do 
		for k in 0.01 0.1
		do 
			imp_dir=der.${k}.ps
			cdir=${exp_dir}/der${i}${k}.imp.${imp_type}
			mkdir -p ${cdir}
			ln -sf ${root}/scripts/init.sh ${cdir}
			ln -sf ${root}/scripts/grid-imp.sh ${cdir}/grid.sh
			ln -sf ${root}/scripts/do-test-imp.sh ${cdir}/do-test.sh
			ln -sf ${root}/scripts/select_params.sh ${cdir}
			ln -sf ${root}/tfboys/tfboys-complex/train ${cdir}/train
			ln -sf ${root}/${data_path}/der${i}${k}.imp/*gt*ffm ${cdir}
			ln -sf ${root}/${data_path}/der${i}${k}.imp/item.ffm ${cdir}
			for j in 'trva' 'tr'
			do
				cat ${root}/${data_path}/der${i}${k}.imp/select_*_${j}.ffm > ${cdir}/${j}.ffm
			done
			for j in 'trva' 'tr'
			do
				ln -sf ${root}/${data_path}/der${i}${k}.imp/select_st_${j}.ffm ${cdir}/imp_${j}.ffm
			done
	
			ln -sf ${root}/${data_path}/der${i}${k}.imp/select_va.ffm ${cdir}/va.ffm
			run_exp_imp ${cdir} ${root} ${mode} ${imp_type} ${root}/${exp_dir}/${imp_dir} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
	
		done
	done
done

## Check command
#echo "Check command list (the command may not be runned!!)"
#task
#wait


# Run
#echo "Run"
#task | xargs -0 -d '\n' -P 4 -I {} sh -c {} 
