#!/bin/bash

. ./init.sh
root=`pwd`
mkdir -p toy_exps_res

for i in "toy-example"
do
	cd ${root}/data
	rm -r ${i}
	cp -r toy-data $i
	wait
	./ob_data.sh ${i}

	cd ${root}/run_dl_exp
	./exp.sh ${i}
	wait

	cd ${root}/run_fm_exp
	./exp.sh ${i}
	wait

	cd ${root}
	python gather_record.py ${i} auc | tee toy_exps_res/${i}.res
	python ratio_analysis.py ${i} toy_exps_res
done
