#!/bin/bash
clear="\033c"
yellow='\033[93m'
end='\033[0m'

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

echo -e ${clear} > /dev/tty1
echo -e "\n${yellow}**Cleaning Up directory...${end}" > /dev/tty1
echo -e "\n${yellow}**Cleaning Up directory...${end}"
rm _dir/*
python aws_bucket.py &
BGPID2=$!
python gpiostates.py 2>&1 > /dev/tty1&
BGPID=$!
echo -e "\nChecking for Button & Audio ..." > /dev/tty1
echo -e "\nChecking for Button & Audio ..."
while true
do
    if [ ! -e _dir/aws_lock ] && [ ! -e _dir/check ] && [ -e _dir/gpio_on ]; then
        rm _dir/tmp.wav
        echo "***** [RECORDING] *****" > /dev/tty1
        echo "***** [RECORDING] *****"
        eval $"rec -c 1 -r 48000 '_dir/tmp.wav' silence 1 0.1 5% 1 0:00:01 1% 2>&1 | cat - > /dev/tty1"
        touch _dir/check
        cp _dir/tmp.wav /media/usb1
        echo "***** [RECORDING ENDED] *****" > /dev/tty1
        echo "***** [RECORDING ENDED] *****"
    fi
    sleep 0.1
done
