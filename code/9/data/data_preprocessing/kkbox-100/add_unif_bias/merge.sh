# Config
rd='rand'
pr='st'
de='det'

filter_data='fl.ffm'
awk '{$1=""; print $0}' $filter_data > $filter_data.tmp &
wait

paste $rd.pos.bias $filter_data.tmp > random.pos.ffm & 
paste $pr.pos.bias $filter_data.tmp > prop.pos.ffm &
paste $de.pos.bias $filter_data.tmp > det.pos.ffm &
wait
rm $filter_data.tmp
rm *.bias

