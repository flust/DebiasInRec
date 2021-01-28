#! /bin/bash

# Source blas
source source-blas.sh

# ==================== #

# Split data 
(cd yh.data/split-data/ &&  ./split.sh)

# Filter data
(cd yh.data/filter.2.5 && ./do-filter.sh)

# Convert-format
(cd yh.data/convert-format && ./setting.sh)

# ==================== #

# Link neccessary files
(cd preprocess && ./init.sh)

# calc-IPS
(cd preprocess/calc-IPS && ./calc.sh)

# calc itemwise r
(cd preprocess/calc-item-r && ./calc.sh)
(cd preprocess/calc-item-r && ./bias.sh)

# calc-r
(cd preprocess/calc-r && ./calc.sh)
(cd preprocess/calc-r && ./bias.sh)

# save imputation model
(cd preprocess/save-imputation-model && make clean all && ./save-model.sh)
(cd preprocess/save-biased-imputation-model && make clean all && ./save-model.sh)

# ==================== #

# Link neccessary files
(cd code && ./init.sh)

# Make 
make -C code/CauseE clean all&
make -C code/FFM-Sc clean all&
make -C code/FFM-St clean all&
make -C code/FFM-Sc_St clean all&
make -C code/IPS clean all&
make -C code/new-complex clean all&
make -C code/new-item-r clean all&
make -C code/new-r clean all&
make -C code/new-complex-bias clean all&
make -C code/new-item-r-bias clean all&
make -C code/new-r-bias clean all&
wait

# CausE
(cd code/CauseE && ./do-test.sh)

# FFM-Sc
(cd code/FFM-Sc && ./do-test.sh)

# FFM-St
(cd code/FFM-St && ./do-test.sh)

# FFM-Sc_St
(cd code/FFM-Sc_St && ./do-test.sh)

# IPS
(cd code/IPS && ./do-test.sh)

# new-complex
(cd code/new-complex && ./do-test.sh)

# new item r
(cd code/new-item-r && ./do-test.sh)

# new r
(cd code/new-r && ./do-test.sh)

# new-complex
(cd code/new-complex-bias && ./do-test.sh)

# new item r
(cd code/new-item-r-bias && ./do-test.sh)

# new r
(cd code/new-r-bias && ./do-test.sh)

# ==================== #

show-test(){
for folder in FFM-St FFM-Sc FFM-Sc_St IPS CauseE new-complex new-item-r new-r new-complex-bias new-item-r-bias new-r-bias
do
    echo '===================='
    echo $folder
    tail -n2 code/$folder/test-logs/* | awk '{ if(NR == 1)  printf "logloss: %s ", $NF; else printf "auc: %s\n", $NF}' 
done
echo '===================='
}

show-test | less
