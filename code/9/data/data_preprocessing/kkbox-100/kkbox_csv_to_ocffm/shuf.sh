#! /bin/bash

out_f="${1/csv/shuf.csv}"

echo $out_f

head -n 1 $1 > $out_f
tail -n +2 $1 | shuf >> $out_f
