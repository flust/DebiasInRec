#!/bin/bash
ln -s ../split-data/yh.f.tr tr.99
ln -s ../split-data/yh.bayes tr.1
ln -s ../split-data/yh.va va.1
ln -s ../split-data/yh.te te.1

cat tr.99 tr.1 > tr.100
cat tr.1 va.1 > trva.1
cat tr.99 va.1 > trva.99
cat tr.100 va.1 > trva.100

for f in tr.* va.* te.* trva.*
do
    python trans.py $f &
done

wait
mkdir -p remap
mv *.remap remap

cd remap
rm item
for i in {0..999} 
do
    echo "0:$i:1" >> item
done

