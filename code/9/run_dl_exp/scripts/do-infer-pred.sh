# Init
#source activate py3.7

# params 
gpu=$1
mode=$2
ps=$3
va_prefix="rnd_gt"
root="test-score.${mode}"
model_path=`find ${root} -name "*.pt"`
#model_name=`echo ${model_path} | cut -d'/' -f2 | cut -d'_' -f1`
model_name=`echo $(basename ${model_path}) | cut -d'_' -f1`
echo "model_name: ${model_name}, model_path: ${model_path}"

task() {
for i in '.' 
do
	for j in '' 'const.' 'pos.'
	do
		va="${va_prefix}${i}${j}"
		va=`echo ${va} | rev | cut -c 2- | rev`
		cmd="python ../../../main.py"
		cmd="${cmd} --dataset_name pos"
		cmd="${cmd} --train_part trva"
		cmd="${cmd} --valid_part ${va}" 
		cmd="${cmd} --dataset_path ./"
		cmd="${cmd} --flag test_auc"
		cmd="${cmd} --model_name ${model_name}"
		cmd="${cmd} --model_path ${model_path}"
		cmd="${cmd} --device cuda:${gpu}"
		cmd="${cmd} --batch_size 1024"
		cmd="${cmd} --ps ${ps}"
		echo "${cmd} > ${root}/${va_prefix}${i}${j}log"
	done
done
}

# Check command
echo "Check command list (the command may not be runned!!)"
task
wait

# Run
echo "Run"
task | xargs -0 -d '\n' -P 1 -I {} sh -c {} 

