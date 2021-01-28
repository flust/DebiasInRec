#!/bin/bash

# Modify 'intel_mkl' variable to your intel mkl path
intel_mkl='/home/user1/intel'

source ${intel_mkl}/mkl/bin/mklvars.sh intel64
MKLPATH=$MKLROOT/lib/intel64_lin
MKLINCLUDE=$MKLROOT/include
echo "Init complete"
echo $MKLROOT
echo $MKLPATH
echo $MKLINCLUDE

