root=$1
mode=$2

for i in 'cbias' 'normal'
do
	for j in 'der.comb.0.01.ffm.wops' 'der.comb.0.1.ffm.wops' 'derive.det.biffm.wops' 'derive.det.extffm.wops' 'derive.det.ffm.wops' 'derive.random.ffm.wops' 
	do
		cdir=${root}/${i}/${j}
		python get_record.py ${cdir} ${mode} > ${i}.record
	done
done

paste *.record | column -s $' ' -t
rm *.record

