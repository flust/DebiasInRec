#! /bin/bash

./shuf.sh train.csv

which python

python filter-by-artist.py
python context_csv_to_ffm.py
python item_csv_to_ffm.py
