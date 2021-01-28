#! /bin/bash

ln -s ../split-data/yh.tr ./yh.all.tr.ps

(cd ocmf && make && ln -s ../yh.all.tr.ps . && ./filter.sh)
cp ocmf/yh.f.tr ../split-data/
