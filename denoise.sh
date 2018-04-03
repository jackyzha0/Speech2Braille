#!/bin/bash

# Please install 'libav-tools' for Ubuntu 14.04 onwards
# As ffmpeg is obsolete, it has been replaced by 'avconv'
# Instead of python script, a simple bash one is sufficient

# A beta release for noNoise-v2

# Usage example
# $ bash noNoise.sh noisyVideo.mp4 noise-reduction-factor

# noise-reduction-factor: 0 means no reduction, 1 means
# maximum damping of noise (recommended is 0.2 to 0.4)

if [ -z $1 ] || [ ! -e $1 ]; then
	echo -e "\nValid sound file needed as args!\n"
	exit 1
fi

# Extracting audio from noisyVideo
noise_file=/tmp/noisy.wav
noisefree_file=/tmp/noisefree.wav
if [ -e ${noise_file} ]; then
	rm -rf ${noise_file}
fi
avconv -loglevel 0 -i $1 -f wav -ab 192000 -vn ${noise_file}

# Creating a noise profile, basically looking for white noise
# in 0 to 0.5 sec of the clip (change if you like)
sox ${noise_file} -n trim 0 -1 noiseprof noisemask

# Removing noise using noise profile
sox ${noise_file} ${noisefree_file} noisered noisemask $2

b_name=$(basename $1)
cp ${noisefree_file} $1
# Replacing noisyAudio with noisefree audio in original video
#avconv  -loglevel 0  -i $1 -i /tmp/noisefree.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 noisefree_$1.mp4
