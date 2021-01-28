#! /bin/bash
l=0.1
t=34
tr=yh.all.tr.ps
./train -l $l -w 0 -a 0 -k 80 -t $t -c 5 -p $tr $tr > yh.f.tr
