nsound

Creating a BLSTM CTC network in Tensorflow for phoneme level speech recognition. More specifically study how adding extra white noise or other sounds (Urban8k or ESC-50) to the background of the TIMIT samples affects the Label Error Rate


Files:
tf_model.py
data_util.py

TODO
-By frame batching
	-20 ms frames
	-10 ms overlap
-Add gaussian noise of 0.6 stddev for better generalization
-Implement Connectionist Temporal Classification as
an optimization function
-Add blank '_\n' for zeroes duration
-Complete namescopes for Tensorflow
-Implement Beam Search Decoder
