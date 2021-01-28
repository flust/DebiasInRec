#!/bin/bash

src=../../yh.data/convert-format/remap
src_pre=../../preprocess

# For CauseE
des=CauseE
ln -s ${src}/item ${des}/

ln -s ${src}/tr.99.remap ${des}/
ln -s ${src}/tr.1.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.99.remap ${des}/
ln -s ${src}/trva.1.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

# For IPS 
des=IPS
ln -s ${src}/item ${des}/

ln -s ${src_pre}/calc-IPS/tr.100.remap.ips ${des}/
ln -s ${src_pre}/calc-IPS/va.1.remap.ips ${des}/

ln -s ${src_pre}/calc-IPS/trva.100.remap.ips ${des}/
ln -s ${src_pre}/calc-IPS/te.1.remap.ips ${des}/

# For new-complex
des=new-complex
ln -s ${src}/item ${des}/

ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

ln -s ${src_pre}/save-imputation-model/tr.100.remap.model ${des}/tr.model
ln -s ${src_pre}/save-imputation-model/trva.100.remap.model ${des}/trva.model

# For new-complex (bias)
des=new-complex-bias
ln -s ${src}/item ${des}/

ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

ln -s ${src_pre}/save-biased-imputation-model/tr.100.remap.model ${des}/tr.model
ln -s ${src_pre}/save-biased-imputation-model/trva.100.remap.model ${des}/trva.model

# For new-item-r
des=new-item-r
ln -s ${src}/item ${des}/

ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

ln -s ${src_pre}/calc-item-r/tr.1.remap.r_score ${des}/tr.r_score

# For new-item-r (bias)
des=new-item-r-bias
ln -s ${src}/item ${des}/

ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

ln -s ${src_pre}/calc-item-r/tr.100.remap.r_score ${des}/tr.r_score

# For new-r
des=new-r
ln -s ${src}/item ${des}/

ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

# For new-r (bias)
des=new-r-bias
ln -s ${src}/item ${des}/

ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

# For FFM-Sc
des=FFM-Sc
ln -s ${src}/item ${des}/

ln -s ${src}/tr.99.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.99.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

# For FFM-St
des=FFM-St
ln -s ${src}/item ${des}/

ln -s ${src}/tr.1.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.1.remap ${des}/
ln -s ${src}/te.1.remap ${des}/

# For FFM-Sc_St
des=FFM-Sc_St
ln -s ${src}/item ${des}/

ln -s ${src}/tr.100.remap ${des}/
ln -s ${src}/va.1.remap ${des}/

ln -s ${src}/trva.100.remap ${des}/
ln -s ${src}/te.1.remap ${des}/
