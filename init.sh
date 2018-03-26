#!/bin/bash

trap ctrl_c SIGINT

function ctrl_c() {
    echo "** Cleaning up and exiting..."
    kill $BGPID
    kill $BGPID2
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
python gpiostates.py &
BGPID2=$!
sleep 40
echo "Checking for Audio..."
while true
do
    if [ ! -e _dir/check ] && [ -e _dir/gpio_on ]; then
        rm _dir/tmp.wav
        echo "***** [RECORDING] *****"
        eval $"rec -c 1 -r 16000 '_dir/tmp.wav' silence 1 0.1 5% 1 0:00:01 3%"
        touch _dir/check
        echo "***** [RECORDING ENDED] *****"
    fi
done
