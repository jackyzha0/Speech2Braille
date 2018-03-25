Speech2Braille - Jacky Zhao

Over 360 million people in the world have disabling hearing loss. Hearing loss can have debilitating effects on a person that makes day to day communication and life very difficult. They face discrimination, as many people and employers find it too much effort to communicate with the deaf. More importantly, the deaf are not able to receive public announcements, warnings, and alarms, which can serve to be a health and safety hazard. For many, current solutions that allow them alleviate these problems have problems themselves that prevent them from being accessible to everyone.

This project entailed creating an end-to-end speech recognition system using an ANN and a portable device to display braille. The device, made from a Raspberry Pi B, is able to recognize audio and transcribe it into Braille through the haptic feedback device via the ANN. The feedback device is a self-made hat, consisting of 6 solenoids, allows the Raspberry Pi to control the 6 solenoids via the GPIO outputs. The neural network itself is 2 layered LSTM network with 256 hidden cells in each layer. The model was trained on the LibriSpeech ‘clean-100’ dataset for 44 hours and 10 minutes, going through 37 epoches, and attained a final accuracy of 74.77% on the training set and 71.50% on the test set. Accuracy of the network was determined using a metric called the Levenshtein edit distance. The model accuracy could not be improved due to time and hardware constraints.


Network Details:
* Character Wise Decoding
* Training Data - LibriSpeech "clean-100"
* Preprocessing for features - 13 MFCCs + 1st Derivatives, normalize to stddev of 1
* Optimizer: RMSPropOptimizer
* Learning Rate: 1e-4
* Number of Hidden Cells: 256
* Number of Layers: 2
* Regulurization: White Noise (stddev: 0.01)
* CTC Loss
* Greedy Decoder
