# Config
rd='random_filter.label'
de='determined_filter.label'
gr='greedy_random_filter.label'
rg='random_greedy_filter.label'
filter_data='filter.ffm'

awk '{$1=""; print $0}' $filter_data > $filter_data.tmp &

merge_label_with_feature(){
    label_file=$1
    output_file_name=$2
    echo "Convert ${label_file} to ${output_file_name}"
    sed -i 's/,$//g' ${label_file}
    paste ${label_file} $filter_data.tmp > ${output_file_name}
}

merge_label_with_feature ${rd} random.ffm &
merge_label_with_feature ${de} det.ffm &
merge_label_with_feature ${gr} greedy_random.ffm &
merge_label_with_feature ${rg} random_greedy.ffm &
wait
echo "Finish converting files."

rm $filter_data.tmp
