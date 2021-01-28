#!/bin/bash

# set up mkl
machine_name=`hostname`
if [ "$machine_name" = "peanuts" ]; then
  echo "$machine_name"
  source /home/skypole/intel/mkl/bin/mklvars.sh intel64
elif [ "$machine_name" = "optima" ]; then
  echo "$machine_name"
  source /home/ybw/intel/mkl/bin/mklvars.sh intel64
elif [ "$machine_name" = "patty" ]; then
  echo "$machine_name"
  source /home/johncreed/intel/mkl/bin/mklvars.sh intel64
elif [ "$machine_name" = "daniel-WS" ]; then
  echo "$machine_name"
  source /home/daniel/Data/intel/mkl/bin/mklvars.sh intel64
elif [[ "$machine_name" =~ linux[0-9]* ]]; then
  echo "$machine_name"
  source /tmp2/d08944012/intel/mkl/bin/mklvars.sh intel64
else
  echo "Not found"
fi
MKLPATH=$MKLROOT/lib/intel64_lin
MKLINCLUDE=$MKLROOT/include
echo "Init complete"
echo $MKLROOT
echo $MKLPATH
echo $MKLINCLUDE

