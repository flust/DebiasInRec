./exp_b2.sh ../data/$1/normal auc 0.5
./exp_b1.sh ../data/$1/normal auc 0.5
./exp_b1_pspw.sh ../data/$1/normal/ auc 0.5
./exp_b1_select.sh ../data/$1/DR auc 0.5
./exp_b1_select.sh ../data/$1/RD auc 0.5
mkdir -p $1
mv DR RD normal $1
