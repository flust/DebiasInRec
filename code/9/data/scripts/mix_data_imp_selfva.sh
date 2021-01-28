#!/bin/bash

root=$1
rnd_ratio=$2

set -x
python sc_st_split.py $1 $2

#trva_num=`wc -l select_st_trva.ffm`
#tr_num=`wc -l select_st_tr.ffm`
#va_num=$(($trva_num-$tr_num))
#tail -n ${va_num} select_st_trva.ffm > select_va.ffm
ln -sf ../der.${rnd_ratio}/select_va.ffm ./
ln -sf ${root}/*gt*ffm ./
ln -sf ${root}/item.ffm ./
