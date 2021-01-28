#!/bin/bash

k=16
t=100
r=0
w=0
wn=1
ns='--ns'
c=5

tr='tr.1.remap'
te='va.1.remap'
item='item'

logs_pth='logs'
mkdir -p $logs_pth

task(){
for k in 16 32 64
do
    for l in 1 4 16
    do
      echo "./train -k $k -l $l -t ${t} -r $r -w $w -wn $wn $ns -c ${c} -p ${te} ${item} ${tr} > $logs_pth/${tr}.$l.$r.$w.$wn.$k"
    done
done
}


task
#task | xargs -d '\n' -P $num_core -I {} sh -c {} &


