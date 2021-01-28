#!/bin/bash

data_path=$1
pos_bias=$2
gpu=$3
mode=$4
ds=$5
ps='wops'

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] ; then
	echo "Plz input: data_path & pos_bias & gpu_idx & mode!!!!!"
	exit 0
fi

root=`pwd`

run_exp(){
	cdir=$1
	rdir=$2
	mode=$3
	model_name=$4
	cmd="cd ${cdir}"
	cmd="${cmd}; ./do-infer.sh ${gpu} ${mode} ${model_name} ${ps} ${best_params}"
	cmd="${cmd}; ./do-infer-pred.sh ${gpu} ${mode} ${ps}"
	cmd="${cmd}; echo 'logloss auc' > ${mode}.record"
	cmd="${cmd}; head -n10 test-score.${mode}/rnd*log >> ${mode}.record" # 
	cmd="${cmd}; cd ${rdir}"
	echo ${cmd}
}

set -e
exp_dir=`basename ${data_path}`
exp_dir="${ds}/${exp_dir}"

for mn in 'biffm' 'extffm'
do
	for i in 'det'
	do
		cdir=${exp_dir}/derive.${i}.${mn}.${ps}
		mkdir -p ${cdir}
		ln -sf ${root}/scripts/*.sh ${cdir}
		ln -sf ${root}/scripts/*.py ${cdir}
		ln -sf ${root}/${data_path}/derive/*gt*svm ${cdir}
		ln -sf ${root}/${data_path}/derive/item.svm ${cdir}
		ln -sf ${root}/${data_path}/derive/truth.svm ${cdir}
		for j in 'trva' 'tr' 'va'
		do
			ln -sf ${root}/${data_path}/derive/${i}_${j}.svm ${cdir}/${j}.svm
		done
		best_params=`grep -nHR "${cdir}" ./best_params | cut -d":" -f4`
		run_exp ${cdir} ${root} ${mode} ${mn} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
	done
done

for mn in 'ffm'
do
	for i in 'random' 'det'
	do
		cdir=${exp_dir}/derive.${i}.${mn}.${ps}
		mkdir -p ${cdir}
		ln -sf ${root}/scripts/*.sh ${cdir}
		ln -sf ${root}/scripts/*.py ${cdir}
		ln -sf ${root}/${data_path}/derive/*gt*svm ${cdir}
		ln -sf ${root}/${data_path}/derive/item.svm ${cdir}
		ln -sf ${root}/${data_path}/derive/truth.svm ${cdir}
		for j in 'trva' 'tr' 'va'
		do
			ln -sf ${root}/${data_path}/derive/${i}_${j}.svm ${cdir}/${j}.svm
		done
		best_params=`grep -nHR "${cdir}" ./best_params | cut -d":" -f4`
		run_exp ${cdir} ${root} ${mode} ${mn} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
	done
done

mn='ffm'
for i in '.comb.' 
do 
	for k in 0.01 0.1
	do 
		cdir=${exp_dir}/der${i}${k}.${mn}.${ps}
		mkdir -p ${cdir}
		ln -sf ${root}/scripts/*.sh ${cdir}
		ln -sf ${root}/scripts/*.py ${cdir}
		ln -sf ${root}/${data_path}/derive/*gt*svm ${cdir}
		ln -sf ${root}/${data_path}/der${i}${k}/item.svm ${cdir}
		ln -sf ${root}/${data_path}/der${i}${k}/truth.svm ${cdir}
		for j in 'trva' 'tr' 'va'
		do
			ln -sf ${root}/${data_path}/der${i}${k}/select_${j}.svm ${cdir}/${j}.svm
		done
		best_params=`grep -nHR "${cdir}" ./best_params | cut -d":" -f4`
		run_exp ${cdir} ${root} ${mode} ${mn} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
	done
done
