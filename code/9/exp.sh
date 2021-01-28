#!/bin/bash

. ./init.sh
root=`pwd`
mkdir -p exps_res

for i in "kkbox-100" "ob-300"
do
	cd ${root}/data/data_preprocessing/$i
	./preprocess.sh

	cd ${root}/data
	rm -r ${i}
	cp -rL data_preprocessing/$i/ocffm-to-ocsvm $i
	wait
	./ob_data.sh ${i}

	cd ${root}/run_dl_exp
	./exp.sh ${i}
	wait

	cd ${root}/run_fm_exp
	./exp.sh ${i}
	wait

	cd ${root}
	python gather_record.py ${i} auc | tee exps_res/${i}.res
	python ratio_analysis.py ${i} exps_res
done
