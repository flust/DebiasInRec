#!/bin/bash

source init.sh

w_train=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8)
l_train=(1 4 16)
t=50
d=32
c=5

item=$1
tr=$2
te=$3
imp=$4

echo "item: $item; tr: $tr; te:$te; imp: $imp"
logs_path="logs/${tr}.${te}"
mkdir -p $logs_path


task(){
for w in ${w_train[@]} 
do
    for l in ${l_train[@]}
    do
        echo "./train -d $d -l $l -t ${t} -w $w --imp-r -c ${c} -p ${te} ${item} ${tr} $imp > $logs_path/$l.$w"
    done
done
}

task
task | xargs -d '\n' -P 15 -I {} sh -c {} &
