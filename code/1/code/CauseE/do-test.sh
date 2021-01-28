#!/bin/bash

ns='--ns'

tr='trva.99.remap'
treat='trva.1.remap'
te='te.1.remap'
item='item'

logs_pth='test-logs'
mkdir -p $logs_pth

task(){
k=32
l=1e-5
ldiff=1e-4
t=50

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
}

task
task | xargs -d '\n' -P 5 -I {} sh -c {} 
