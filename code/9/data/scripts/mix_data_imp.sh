#!/bin/bash

root=$1
rnd_ratio=$2

set -x
python sc_st_split.py $1 $2

ln -sf ${root}/random_va.ffm select_va.ffm
ln -sf ${root}/*gt*ffm ./
ln -sf ${root}/item.ffm ./
