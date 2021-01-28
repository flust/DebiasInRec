#!/bin/bash

source init.sh

l=16
r=-6
w=0.03125

t=4

./train -k 8 -l $l -t $t -r $r -w $w -wn 1  -c 5 -o filter.model item.ffm rd.trva.ffm 

echo "./train -k 8 -l $l -t $t -r $r -w $w -wn 1  -c 5 -o filter.model item.ffm rd.trva.ffm" > filter.model.param
