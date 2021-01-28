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

w_train=(1.0 0.0625 0.00390625 0.000244140625 1.52587890625e-05)
l_train=(`awk -v x=${imp_l} -v y=4.0 'BEGIN {printf "%g\n",x/y}'` ${imp_l})
t=50
d=${imp_k}
c=8

item='item.ffm'
tr='tr.ffm'
te='va.ffm'
imp='imp_tr.ffm'

#echo "item: $item; tr: $tr; te:$te; imp: $imp"
logs_path="logs"
mkdir -p $logs_path


task(){
for w in ${w_train[@]} 
do
    for l in ${l_train[@]}
    do
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
		cmd="${cmd} -p ${te} ${item} ${tr} ${imp} > $logs_path/${l}_${w}_${d}"
		echo "${cmd}"
    done
done
}

task
task | xargs -d '\n' -P 3 -I {} sh -c {} #rm Pva* Qva*
