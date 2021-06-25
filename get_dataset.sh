#!/bin/sh

cd ./preprocess

# download dataset
wget -c https://www.egr.msu.edu/denseleaves/Data/DenseLeaves.zip

# unzip
unzip -o ./DenseLeaves.zip

python ./generate_dataset.py
cd ..
echo "DONE"