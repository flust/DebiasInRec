#!/bin/bash

data='context.ffm'

random_data='rd.trva.ffm'
filter_data='filter.ffm'
truth_data='truth.ffm'

shuf ${data}  > ${data}.shuf
data=${data}.shuf

total_num=`wc -l ${data} | awk '{print $1}'`
small_perc_total_num=`echo "scale=0;$total_num*10/100" | bc -l `
large_perc_total_num=`echo "scale=0;$total_num*80/100" | bc -l `
echo $total_num $small_perc_total_num $large_perc_total_num


head -n $small_perc_total_num ${data} > $random_data &
head -n $(($small_perc_total_num + $large_perc_total_num)) ${data} | tail -n $large_perc_total_num > $filter_data &
tail -n $(($total_num - $small_perc_total_num - $large_perc_total_num)) ${data} > $truth_data &

wait

half_small_perc_total_num=`echo "scale=0;$small_perc_total_num/2" | bc -l `
head -n $half_small_perc_total_num $random_data > rd.tr.ffm &
tail -n $(($small_perc_total_num - $half_small_perc_total_num)) $random_data > rd.va.ffm &
