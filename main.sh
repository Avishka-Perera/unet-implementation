#! /bin/bash

if [ ! -d "out" ]; then
    mkdir -p "out"
fi

nohup bash expm/isbi-em/main.sh > out/isbi-em.log &
sleep 5 # avoid crashing due to port
nohup bash expm/isbi-cell_track-dic_hela/main.sh > out/isbi-cell_track-dic_hela.log &
sleep 5 # avoid crashing due to port
nohup bash expm/isbi-cell_track-phc_u373/main.sh > out/isbi-cell_track-phc_u373.log &