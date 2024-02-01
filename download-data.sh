#! /bin/bash

if [ ! -d "datasets" ]; then
    mkdir -p "datasets"
fi

cd datasets

wget http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip # 41M
wget http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip # 40M
wget https://github.com/hoangp/isbi-datasets/archive/refs/heads/master.zip # 4.97M

unzip PhC-C2DH-U373.zip
unzip DIC-C2DH-HeLa.zip 
unzip master.zip

rm PhC-C2DH-U373.zip
rm DIC-C2DH-HeLa.zip 
rm master.zip

mv isbi-datasets-master/data ISBI-2012-challenge
rm -r isbi-datasets-master

cd ..