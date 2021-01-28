#!/bin/sh
set -e

root=$1
mode=$2

if [ "${mode}" = "auc" ]; then
	col=4
	opt=''
else
	col=3
	opt='-r'
fi

echo 'stats' > ${mode}.stats
for i in `ls ${root}`
do
	#cat ${root}/${i} | awk '{if (NF==4 && NR>2) print $0}' | sort -gk${col} ${opt} | tail -n1 | xargs -I {} grep '{}' ${root}/* >> ${mode}.stats #| awk -F ' ' '{print $4}'
	cat ${root}/${i} | awk '{if (NF==4 && NR>2) print $0}' | sort -gk${col} ${opt} | tail -n1 | xargs -I {} echo "${i}:{}"  >> ${mode}.stats #| awk -F ' ' '{print $4}'
done

cat ${mode}.stats | awk '{if (NR>1) print $0}' | sort -gk${col} ${opt} | tail -n1 > ${mode}.top
#rm ${mode}.stats


