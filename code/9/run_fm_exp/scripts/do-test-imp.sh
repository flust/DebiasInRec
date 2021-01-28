#!/bin/bash
. ./init.sh

imp_type=$1
imp_dir=$2
mode=$3

root=`pwd`
if [ -d "${imp_dir}" ]; then
	cd ${imp_dir}
	imp_t=`python select_params.py logs ${mode} | cut -d' ' -f3`
	imp_l=`python select_params.py logs ${mode} | cut -d' ' -f1`
	imp_k=`python select_params.py logs ${mode} | cut -d' ' -f2`
	cd ${root}
else
	echo "Can not find dir ${imp_dir}!"
	exit 0
fi

./select_params.sh logs ${mode}
l=`cat ${mode}.top | awk -F ':' '{print $1}' | rev | awk -F '/' '{print $1}'| rev | awk -F '_' '{print $1}'`
w=`cat ${mode}.top | awk -F ':' '{print $1}' | rev | awk -F '/' '{print $1}'| rev | awk -F '_' '{print $2}'`
d=`cat ${mode}.top | awk -F ':' '{print $1}' | rev | awk -F '/' '{print $1}'| rev | awk -F '_' '{print $3}'`
t=`cat ${mode}.top | awk -F ':' '{print $2}' | awk -F ' ' '{print $1}'`
c=8

item='item.ffm'
tr='trva.ffm'
te_prefix='rnd_gt'
imp='imp_trva.ffm'

#echo "item: $item; tr: $tr; te:$te; imp: $imp"

logs_path="test-score.${mode}"
mkdir -p $logs_path; 

task(){
for i in '.' 
do
	for j in '' 'const.' 'pos.'
	do
		te="${te_prefix}${i}${j}ffm"
		cmd="./train"
		cmd="${cmd} -l $l"
		cmd="${cmd} -w $w"
		cmd="${cmd} -d $d"
		cmd="${cmd} -t ${t}"
		if [ "${imp_type}" == "avg" ]; then
			cmd="${cmd} -imp-r 0"
		else
			cmd="${cmd} -L $imp_l -D $imp_k -T ${imp_t}"
		fi
		cmd="${cmd} -c ${c}"
		#cmd="${cmd} --save-model"
		cmd="${cmd} -p ./${te} ./${item} ./${tr} ./${imp} > ${logs_path}/${l}_${w}_${d}${i}${j}log"
		echo "${cmd}"
	done
done
}

task
task | xargs -d '\n' -P 3 -I {} sh -c {} 
