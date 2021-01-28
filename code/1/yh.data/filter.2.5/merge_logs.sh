#!/bin/bash

for l in 1 0.1 0.01 0.001 0.0001
do
    cd logs.${l}.k80
    python ../merge.py $l
    sort -gk 2 yh.all.tr.ps.te.${l}.k80.std.merge | head -n 1 > ../${l}.best_score 
    cd ..
done 
