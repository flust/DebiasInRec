#!/bin/bash

wn=1
ns='--ns'

tr='trva.100.remap'
te='te.1.remap'
item='item'
item_r='tr.r_score'

logs_pth='test-logs'
mkdir -p $logs_pth

task(){
t=4
k=64
l=2
w=0.00001525878906
            
cmd='./train'
cmd="$cmd -k $k"
cmd="$cmd -l $l"
cmd="$cmd -t $t"
cmd="$cmd -w $w"
cmd="$cmd -c 5"
cmd="$cmd -wn ${wn}"
cmd="$cmd -item-r ${item_r}"
cmd="$cmd $ns"
cmd="$cmd -p ${te}"
cmd="$cmd ${item} ${tr}"
cmd="$cmd > $logs_pth/${tr}.${k}.${l}.${w}.$ns"
echo $cmd
}

task
task | xargs -d '\n' -P 5 -I {} sh -c {} 
