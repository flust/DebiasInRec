#! /bin/bash

ffm_ext=$1
unif_ffm_ext=$2

item="item.ffm"
truth="truth.ffm" # 10% data
filter="filter.ffm" # 80% data

ffm_files="det.${ffm_ext} random.${ffm_ext} 
greedy_random.${ffm_ext} random_greedy.${ffm_ext}"
unif_ffm_files="det.${unif_ffm_ext} random.${unif_ffm_ext} 
greedy_random.${unif_ffm_ext} random_greedy.${unif_ffm_ext}"

# First file must be item.ffm 
python ocffm-to-ocsvm.py ${item} ${filter} ${truth} ${ffm_files} ${unif_ffm_files}
