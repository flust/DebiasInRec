
./exp_ffm.sh ../data/$1/normal 0.5 0 auc
./exp_ffm_b2.sh ../data/$1/normal 0.5 0 auc
./exp_ffm_b2_select.sh ../data/$1/cbias 0.5 0 auc
mkdir -p $1
mv normal cbias $1
