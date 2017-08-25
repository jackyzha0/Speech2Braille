nsound
Repository for machine learning projects
Currently working on hybrid BLSTM-HMM model for robust speech recognition in polyphonic environments

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
