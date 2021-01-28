#!/bin/bash

r=-3.82825
wn=1
ns='--ns'

tr='trva.100.remap'
te='te.1.remap'
item='item'

logs_pth='test-logs'
mkdir -p $logs_pth

task(){
t=18
k=16
l=0.5
w=0.0625

cmd='./train'
cmd="$cmd -k $k"
cmd="$cmd -l $l"
cmd="$cmd -t $t"
cmd="$cmd -w $w"
cmd="$cmd -c 5"
cmd="$cmd -wn ${wn}"
cmd="$cmd -r ${r}"
cmd="$cmd $ns"
cmd="$cmd -p ${te}"
cmd="$cmd ${item} ${tr}"
cmd="$cmd > $logs_pth/${tr}.${k}.${l}.${w}.$ns"
echo $cmd
}

task
task | xargs -d '\n' -P 5 -I {} sh -c {} 
