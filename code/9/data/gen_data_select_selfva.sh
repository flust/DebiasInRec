#!/bin/bash

root=`pwd`
data_path=${1}
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
ln -sf ${root}/scripts/trans_data_select_selfva.sh ./
ln -sf ${root}/scripts/chg_form.py ./
ln -sf ${root}/scripts/gen_rnd_gt.py ./
./trans_data_select_selfva.sh ../origin/ ${pos_bias}
cd ${root}

