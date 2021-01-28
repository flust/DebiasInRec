#! /bin/bash
root=`pwd`
cd hybrid-ocffm
source ${root}/scripts/init.sh
make
cd ..
cd tfboys/tfboys-imp-model
source ${root}/scripts/init.sh
make
cd ../..
cd tfboys/tfboys-complex
source ${root}/scripts/init.sh
make
cd ../..
cd tfboys/tfboys-complex-imp-grid
source ${root}/scripts/init.sh
make
