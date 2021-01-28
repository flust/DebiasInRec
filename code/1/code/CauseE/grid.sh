#!/bin/bash

t=100
ns='--ns'

tr='tr.99.remap'
treat='tr.1.remap'
te='va.1.remap'
item='item'

logs_pth='logs'
mkdir -p $logs_pth

task(){
for k in 8 16 32 64 
do
    for l in 1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6
    do
        for ldiff in 1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 
        do
            cmd='./train'
            cmd="$cmd -k $k"
            cmd="$cmd -l $l"
            cmd="$cmd -ldiff $ldiff"
            cmd="$cmd -t $t"
            cmd="$cmd -c 5"
            cmd="$cmd $ns"
            cmd="$cmd -p ${te}"
            cmd="$cmd --treat ${treat}"
            cmd="$cmd ${item} ${tr}"
            cmd="$cmd > $logs_pth/${tr}.${k}.${l}.${ldiff}.$ns"
            echo $cmd
        done
    done
done
}

task
task | xargs -d '\n' -P 5 -I {} sh -c {} &
