#!/bin/bash

k=16
l=4
t=10

trva=tr.100.remap
./train -l ${l} -t ${t} -p ${trva} -save-imp ${trva}.model -w 0 -wn 1 -r 0 -c 5 -k ${k} --ns item tr.1.remap &

trva=trva.100.remap
./train -l ${l} -t ${t} -p ${trva} -save-imp ${trva}.model -w 0 -wn 1 -r 0 -c 5 -k ${k} --ns item tr.1.remap &
