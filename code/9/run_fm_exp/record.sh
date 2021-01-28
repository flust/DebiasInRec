#! /bin/bash

root=$1
mode=$2

count=0
for i in 'der.0.01' 'der.0.1' 'der.comb.0.01' 'der.comb.0.1' 'derive.det' 'derive.random'
do
    python get_record.py ${root}/${i} ${mode} > ${count}.record
    (( count++ ))
done

for i in 'der.comb.0.01.imp.pw' 'der.comb.0.1.imp.pw'
do
    ./record-imp.sh ${root}/${i} ${mode} > ${count}.record
    (( count++ ))
done

paste *.record | column -s $' ' -t
rm *.record

