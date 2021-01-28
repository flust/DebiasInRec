# Init
#source activate py3.7

# params 
gpu=$1
mode=$2
model_name=$3
ps=$4

# Data set
ds='pos'
tr_part='tr'
va_part='va'
ds_path='./'

# Fixed parameter
flag='train'
epoch=20
bs=512

# others
log_path="logs"
device="cuda:${gpu}"

task(){
# Set up fixed parameter and train command
train_cmd="python ../../main.py"
train_cmd="${train_cmd} --dataset_name ${ds}"
train_cmd="${train_cmd} --train_part ${tr_part}"
train_cmd="${train_cmd} --valid_part ${va_part}"
train_cmd="${train_cmd} --dataset_path ${ds_path}"
train_cmd="${train_cmd} --flag ${flag}"
train_cmd="${train_cmd} --model_name ${model_name}"
train_cmd="${train_cmd} --epoch ${epoch}"
train_cmd="${train_cmd} --device ${device}"
train_cmd="${train_cmd} --save_dir ${log_path}"
train_cmd="${train_cmd} --batch_size ${bs}"
train_cmd="${train_cmd} --ps ${ps}"

# Print out all parameter pair
for lr in 0.001 #0.001 0.0001
do
    for wd in 1e-5 1e-6 1e-7
    do
        for k in 32 
        do
            cmd="${train_cmd} --learning_rate ${lr}"
            cmd="${cmd} --weight_decay ${wd}"
            cmd="${cmd} --embed_dim ${k}"
            echo "${cmd}"
        done
    done
done
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
#wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
