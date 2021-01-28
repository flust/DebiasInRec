set -x

python ../../pred.py ./
for i in 0 1 2 3 4
do
	#python ../../cal_revenue.py test-score/test.pred.${i} gt.ffm 0.5 | tee test-score/${i}.log
	python ../../cal_revenue.py tmp.pred.${i} gt.ffm 0.5 | tee test-score/${i}.log
	#mv tmp.pred.${i} test-score/test.pred.${i}
done 
