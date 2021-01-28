#!/bin/bash

source init.sh

c=5
k=8
t=20
tr=rd.tr.ffm
va=rd.va.ffm
item=item.ffm
logs_pth=logs/${tr}.${va}.${k}

mkdir -p $logs_pth

# 2^0 ~ 2^-11
w_train=(1  0.25  0.0625  0.015625  0.00390625  0.0009765625  0.000244140625  6.103515625e-05  1.52587890625e-05)
wn_train=(1)
l_train=( 4 )
r_train=(-8)

task(){
  for w in ${w_train[@]} 
  do
      for r in ${r_train[@]}
      do
        for l in ${l_train[@]}
        do
          echo "./train -k $k -l $l -t ${t} -r $r -w $w -wn 1 -c ${c} -p ${va} ${item} ${tr} > $logs_pth/${tr}.$l.$r.$w"
        done
      done
  done
}


num_core=4

#task
task | xargs -d '\n' -P $num_core -I {} sh -c {} &
