nsound
Repository for GVRSF 2k18

Abstract:
Over 360 million people have disabling hearing loss. One of the main impacts hearing loss has, is on an individual’s ability to communicate with others. This can have adverse affects on people, such as causing emotional distress, greater need for assistance, and missed opportunities for employment or education. Citing from the WHO, “[The cost of] Loss of productivity, due to unemployment and premature retirement among people with hearing loss, is conservatively estimated to cost $678 billion annually.” Most people with hearing loss are able to communicate in the world through lip reading and other visual cues; however, are unable to react to audio cues where the speaker is not visible (behind them, speakers, alarms, etc.) This debilitates those with hearing loss in the workforce and creates dangerous scenarios in which these people may not hear an alarm.

This experiment entailed creating a machine learning network to convert streams of audio data into a readable output (Braille).

Network Details:
[Input - TIMIT Dataset]
[Preprocessing for features - 13 MFCCs + 12 Chromograms + 7 Bands of Spectral Contrast + 6 Bands of Tonal Centroid
Features + Zero Crossing Rate + Spectral Rolloff + Spectral Centroid]
[Reshaping and batching for input - Rank 3 Tensor (batchsize, maxstepsize, num_features)]
[BLSTM Network Dimensions]
* Learning Rate = 1e-2
* Momentum = 0.9
* Number of Hidden Layers = 4
* Network Depth = 2
[CTC Loss and Decoder]
