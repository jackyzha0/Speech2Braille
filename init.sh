#!/bin/bash

trap ctrl_c SIGINT

function ctrl_c() {
    echo "** Cleaning up and exiting..."
    kill $BGPID
    rm _dir/*
    exit
}
if [ "$EUID" -ne 0 ]
    then echo "Please run as root"
    exit
fi

echo "** Preloading Python Modules..."
python load_meta.py &
BGPID=$!
rm _dir/*
sleep 2
while true
do
    if [ ! -f _dir/check ]; then
        rm _dir/tmp.wav
        echo "***** [RECORDING] *****"
        eval $"rec -c 1 -r 16000 '_dir/tmp.wav' silence 1 0.1 10% 1 0:00:01 3%"
        touch _dir/check
        echo "***** [RECORDING ENDED] *****"
    fi
done
