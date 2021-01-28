#!/bin/bash

root=`pwd`
data_path=$1
pos_bias=$2

if [ -z "$1" ] || [ -z "$2" ]; then
	echo "Plz input data_path & pos_bias!!!!!"
	exit 0
fi

set -ex
# original data
cd ${data_path}
mkdir origin
mv *.bias *.ffm *.svm origin/
cd ${root}

# generate S_c and S_t
cdir=${data_path}/derive
mkdir -p ${cdir} 
cd ${cdir}
ln -sf ${root}/scripts/trans_data.sh ./
ln -sf ${root}/scripts/chg_form.py ./
ln -sf ${root}/scripts/gen_rnd_gt.py ./
./trans_data.sh ../origin/ ${pos_bias}
cd ${root}

# generate 99%S_c + 1%S_t, 90%S_c + 10%S_t, 10%S_t and 1%S_t
for i in '.comb.' '.'
do 
	for j in 0.01 0.1
	do 
		cdir=${data_path}/der${i}${j}
		mkdir -p ${cdir} 
		cd ${cdir}
		ln -sf ${root}/scripts/mix_data.sh ./
		ln -sf ${root}/scripts/gen_mix_data.py ./
		./mix_data.sh ../derive/ ${j} ${i}
		cd ${root}
	done
done

# generate imp data for 99%S_c + 1%S_t and 90%S_c + 10%S_t 
for i in '.comb.' 
do 
	for j in 0.01 0.1
	do 
		cdir=${data_path}/der${i}${j}.imp
		mkdir -p ${cdir} 
		cd ${cdir}
		ln -sf ${root}/scripts/mix_data_imp.sh ./
		ln -sf ${root}/scripts/sc_st_split.py ./
		./mix_data_imp.sh ../derive/ ${j} 
		cd ${root}
	done
done
