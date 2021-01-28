#!/bin/bash

tr=tr.100.remap
python calc-r.py ${tr} > ${tr}.r_score &
