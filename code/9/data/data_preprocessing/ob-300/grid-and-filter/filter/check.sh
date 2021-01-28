#! /bin/bash

check(){
    echo $1
    total_count=0
    for i in {1..10}
    do
        position_count=`awk -F , -v pos=$i '{print $pos}' $1 | grep ':1:' | wc -l`
        total_count=`echo ${total_count} + ${position_count} | bc -l `
        echo ${position_count} ${total_count}
    done

}

check determined_filter.label > de_check.txt &
check random_filter.label > rd_check.txt &
check greedy_random_filter.label > gr_check.txt &
check random_greedy_filter.label > rg_check.txt &
