# Init
#source activate py3.7

# params 
gpu=$1
mode=$2
model_name=$3
ps=$4

# Data set
ds='pos'
tr_part='trva'
va_part='rnd_gt.pos'
ds_path='./'

# Fixed parameter
flag='train'
lr=`python select_params.py logs ${mode} | cut -d' ' -f1`
wd=`python select_params.py logs ${mode} | cut -d' ' -f2`
bs=`python select_params.py logs ${mode} | cut -d' ' -f3` 
k=`python select_params.py logs ${mode} | cut -d' ' -f4`
epoch=`python select_params.py logs ${mode} | cut -d' ' -f5`
epoch=$((${epoch}+1))

# others 
log_path="test-score.${mode}"
device="cuda:${gpu}"

task(){
# Set up fixed parameter and train command
cmd="python ../../main.py"
cmd="${cmd} --dataset_name ${ds}"
cmd="${cmd} --train_part ${tr_part}"
cmd="${cmd} --valid_part ${va_part}"
cmd="${cmd} --dataset_path ${ds_path}"
cmd="${cmd} --flag ${flag}"
cmd="${cmd} --model_name ${model_name}"
cmd="${cmd} --epoch ${epoch}"
cmd="${cmd} --device ${device}"
cmd="${cmd} --save_dir ${log_path}"
cmd="${cmd} --batch_size ${bs}"
cmd="${cmd} --ps ${ps}"
cmd="${cmd} --learning_rate ${lr}"
cmd="${cmd} --weight_decay ${wd}"
cmd="${cmd} --embed_dim ${k}"
echo "${cmd}"
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
#wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
