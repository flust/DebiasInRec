#!/bin/sh

root=$1
pos_bias=$2

set -x
for i in 'det' 'random'
do
	for j in 'ffm' 'svm'
	do
		ln -sf ${root}/${i}.${j}.pos.${pos_bias}.bias ${i}.${j}
	done
done
ln -sf ${root}/truth.* ./
ln -sf ${root}/item.* ./

num_trva=`wc -l random.ffm | cut -d' ' -f1`
num_te=`wc -l truth.svm | cut -d' ' -f1`
num_tr=$((${num_trva} - ${num_te}))
num_item=`wc -l item.ffm | cut -d' ' -f1`

for i in 'det' 'random'
do
	for j in 'ffm' 'svm'
	do
		python chg_form.py ${i}.${j} ${i}
		split -l ${num_tr} ${i}_trva.${j}
		mv xaa ${i}_tr.${j}
		mv xab ${i}_va.${j}
	done
done

for i in 'ffm' 'svm'
do
	rm det_trva.${i} det_va.${i} random_trva.${i}
	ln -sf det_tr.${i} det_trva.${i}
	ln -sf random_tr.${i} random_trva.${i}
	ln -sf random_va.${i} det_va.${i}
	#for c in `seq 1 10`
	#do
	#	cat truth.${i} >> truth.10.${i}
	#done
	for j in '.' '.const.' '.pos.'
	do
		python gen_rnd_gt.py truth.${i} rnd_gt${j}${i} ${num_item} ${j} ${pos_bias} &
		#python gen_rnd_gt.py truth.10.${i} rnd_gt.10${j}${i} ${num_item} ${j} ${pos_bias} &
	done
	wait
done
python chg_form.py truth.ffm truth
mv truth_trva.ffm gt.ffm


