root=`pwd`

cd $1
mkdir RD
mkdir DR
mkdir cbias
mkdir normal
mkdir stats
item_num=`wc -l item.ffm | cut -d" " -f1`
mv random_greedy.* RD
mv greedy_random.* DR
mv *unif* cbias
cp truth.* item.* RD
cp truth.* item.* DR
cp truth.* item.* cbias
mv det.* random.* item.* truth.* normal

cd RD
ln -sf random_greedy.ffm.pos.0.5.bias det.ffm.pos.0.5.bias
ln -sf random_greedy.svm.pos.0.5.bias det.svm.pos.0.5.bias
cd ../

cd DR
ln -sf greedy_random.svm.pos.0.5.bias det.svm.pos.0.5.bias
ln -sf greedy_random.ffm.pos.0.5.bias det.ffm.pos.0.5.bias
cd ../

cd cbias
ln -sf random.ffm.pos.0.5.unif.bias det.ffm.pos.0.5.bias
ln -sf random.svm.pos.0.5.unif.bias det.svm.pos.0.5.bias
cd ../

cd stats
ln -sf ../normal/origin/det.ffm.pos.0.5.bias D.ffm.pos.0.5.bias
ln -sf ../normal/origin/random.ffm.pos.0.5.bias R.ffm.pos.0.5.bias
ln -sf ../RD/origin/random_greedy.ffm.pos.0.5.bias RD.ffm.pos.0.5.bias
ln -sf ../DR/origin/greedy_random.ffm.pos.0.5.bias DR.ffm.pos.0.5.bias
cd ../

cd $root 
./gen_data_selfva.sh $1/normal 0.5
./gen_data_select_selfva.sh $1/RD 0.5
./gen_data_select_selfva.sh $1/DR 0.5
./gen_data_select_selfva.sh $1/cbias 0.5

for i in 'R' 'D' 'DR' 'RD'
do 
	python stats.py $1/stats/$i.ffm.pos.0.5.bias $1/stats/$i.csv $item_num & 
done
wait
echo '================================='
