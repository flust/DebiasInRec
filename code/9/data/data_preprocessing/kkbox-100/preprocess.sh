#! /bin/bash

# convert csv to ocffm
cd kkbox_csv_to_ocffm
./run_filter.sh

# split to G1 G2 G3
cd ../split-data
./split.sh

# train on G1
cd ../grid-and-filter/hyffm-grid
make clean all
# grid model and select best parameter 
# We already save the best parameter into save_model.sh files.
# Thus, we skip the grid process.
# ./grid.sh 
./save_model.sh

# filter G2 by model which learned from G1
cd ../filter
make clean all
./run-filter.sh

# Add position bias
cd ../../add_bias
./add_bias_all.sh

# Add position bias
cd ../add_unif_bias
./add_bias_all.sh

# convert data to svm format
cd ../ocffm-to-ocsvm
./convert.sh ffm.pos.0.5.bias ffm.pos.0.5.unif.bias

cd ..
