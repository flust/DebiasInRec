#!/bin/bash

t=100
wn=1
ns='--ns'

tr='tr.100.remap'
te='va.1.remap'
item='item'
imp='tr.model'

logs_pth='logs'
mkdir -p $logs_pth

task(){
for k in 8 16 32 64 
do
    for l in 1 2 4 8 16 32 64 128 
    do
        for w in 0.00390625 0.0009765625 0.000244140625  
        do
            cmd='./train'
            cmd="$cmd -k $k"
            cmd="$cmd -l $l"
            cmd="$cmd -t $t"
            cmd="$cmd -w $w"
            cmd="$cmd -c 5"
            cmd="$cmd -wn ${wn}"
            cmd="$cmd -imp ${imp}"
            cmd="$cmd $ns"
            cmd="$cmd -p ${te}"
            cmd="$cmd ${item} ${tr}"
            cmd="$cmd > $logs_pth/${tr}.${k}.${l}.${w}.$ns"
            echo $cmd
        done
    done
done
}

task
#task | xargs -d '\n' -P $num_core -I {} sh -c {} &
