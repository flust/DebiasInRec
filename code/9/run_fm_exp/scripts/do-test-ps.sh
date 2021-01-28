#! /bin/bash
. ./init.sh

# Train file path
train='./hytrain'

# params selection
mode=$1

# Fixed parameter
c=8

# Data setr
tr='trva.ffm'
va_prefix='rnd_gt'
item='item.ffm'

# Log path
log_path="test-score.${mode}"
mkdir -p $log_path


task(){

for i in '.' 
do
	for j in '' 'const.' 'pos.'
	do
		va="${va_prefix}${i}${j}ffm"
		# Print out all parameter pair
		echo "python select_params_ps.py logs ${mode} | cut -d' ' -f3"
		t=`python select_params_ps.py logs ${mode} | cut -d' ' -f3`
		l=`python select_params_ps.py logs ${mode} | cut -d' ' -f1`
		d=`python select_params_ps.py logs ${mode} | cut -d' ' -f2`
		train_cmd="${train} -c ${c}"
		cmd=${train_cmd}
		cmd="${cmd} -t ${t}"
		cmd="${cmd} -l ${l}"
		cmd="${cmd} -d ${d}"
		cmd="${cmd} -p ${va} ${item} ${tr} > ${log_path}/$l.0.$d${i}${j}log"
		#echo "${cmd}; mv Pva* Qva* ${log_path}"
		echo "${cmd}"
	done
done
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 3 -I {} sh -c {} 
