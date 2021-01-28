#! /bin/bash
tasks(){
    for f in {0..4}
    do
        te="yh.all.tr.ps.$f.te"
        tr="yh.all.tr.ps.$f.tr"
        for l in 1 0.1 0.01 0.001 0.0001
        do
            mkdir -p logs.$l.k80
            echo "./train -l $l -w 0 -a 0 -k 80 -t 200 -c 5 -p $te $tr > logs.${l}.k80/${te}.${l}.k80.std"
        done
    done
}

tasks | xargs  -P 4 -I {}  sh -c {} &
