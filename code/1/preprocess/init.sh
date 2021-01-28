#!/bin/bash

src=../../yh.data/convert-format/remap

# For IPS
des=calc-IPS
ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/tr.1.remap ${des}/
ln -s ${src}/va.1.remap ${des}/
ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/trva.1.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

# For item-r
des=calc-item-r
ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/tr.1.remap ${des}/

# For calc-r
des=calc-r
ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/tr.1.remap ${des}/

# For grid-and-save-imputation-model 
des=save-imputation-model
ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/tr.1.remap ${des}/
ln -s ${src}/va.1.remap ${des}/
ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/item ${des}/

# For grid-and-save-imputation-model 
des=save-biased-imputation-model
ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/va.1.remap ${des}/
ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/item ${des}/
