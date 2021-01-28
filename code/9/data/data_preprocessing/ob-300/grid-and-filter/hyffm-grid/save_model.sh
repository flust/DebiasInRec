#!/bin/bash

source init.sh

l=4
r=-5
w=0.0625

t=20

./train -k 8 -l $l -t $t -r $r -w $w -wn 1  -c 5 -o filter.model item.ffm rd.trva.ffm 

echo "./train -k 8 -l $l -t $t -r $r -w $w -wn 1  -c 5 -o filter.model item.ffm rd.trva.ffm" > filter.model.param
