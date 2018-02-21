nsound

Abstract:
Over 360 million people have disabling hearing loss. One of the main impacts hearing loss has, is on an individual’s ability to communicate with others. This can have adverse affects on people, such as causing emotional distress, greater need for assistance, and missed opportunities for employment or education. Citing from the WHO, “[The cost of] Loss of productivity, due to unemployment and premature retirement among people with hearing loss, is conservatively estimated to cost $678 billion annually.” Most people with hearing loss are able to communicate in the world through lip reading and other visual cues; however, are unable to react to audio cues where the speaker is not visible (behind them, speakers, alarms, etc.) This debilitates those with hearing loss in the workforce and creates dangerous scenarios in which these people may not hear an alarm.

This experiment entailed creating a machine learning network to convert streams of audio data into a readable output (Braille).

Network Details:
* Character Wise Decoding
* Input - TIMIT Dataset
* Preprocessing for features - 13 MFCCs
* Reshaping and batching for input - Rank 3 Tensor (batchsize, time, num_features)
* Deep LSTM Network Hyperparemters
	* Learning Rate = 1e-3
	* Momentum = 0.9
	* Dropout Rate = 0.5
	* Number of Hidden Layers = 100
	* Network Depth = 2
	* CTC Loss and Decoder
