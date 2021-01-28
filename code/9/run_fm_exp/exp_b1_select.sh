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
	cmd="${cmd}; ./do-test.sh ${mode}"
	cmd="${cmd}; echo 'va_logloss va_auc' > ${mode}.record"
	cmd="${cmd}; python select_params_ps.py logs ${mode} | rev | cut -d' ' -f1-2 | rev >> ${mode}.record"
	cmd="${cmd}; tail -n1 test-score.${mode}/*.log | rev | awk -F' ' '{print \$1, \$2}'| rev >> ${mode}.record"
	cmd="${cmd}; cd ${rdir}"
	echo ${cmd}
}

set -e
exp_dir=`basename ${data_path}`

for i in 'det' 
do
	cdir=${exp_dir}/derive.${i}.select.ps
	mkdir -p ${cdir}
	ln -sf ${root}/scripts/init.sh ${cdir}
	ln -sf ${root}/scripts/grid-ps.sh ${cdir}/grid.sh
	ln -sf ${root}/scripts/do-test-ps.sh ${cdir}/do-test.sh
	ln -sf ${root}/scripts/select_params_ps.py ${cdir}
	ln -sf ${root}/tfboys/tfboys-complex-imp-grid/train ${cdir}/hytrain
	ln -sf ${root}/${data_path}/derive/*gt*ffm ${cdir}
	ln -sf ${root}/${data_path}/derive/item.ffm ${cdir}
	for j in 'trva' 'tr' 'va'
	do
		ln -sf ${root}/${data_path}/derive/${i}_${j}.ffm ${cdir}/${j}.ffm
	done
	run_exp ${cdir} ${root} ${mode} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
done

