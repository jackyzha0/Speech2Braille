import requests
import wave
import json

subscription_key = '3c86100c6a4b482b9a41552d5f05859b'
ENDPOINT = 'https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1'

def transcribe(path):
    with open(path, 'rb') as fobj:
        res = requests.post(url=ENDPOINT,
                            data=fobj,
                            params= {'language': 'en-US',
                                'profanity': 'raw'},
                            headers={'Ocp-Apim-Subscription-Key': subscription_key,
                                'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
                                'Accept': 'application/json;text/xml'}
                            )
    return res.json()['DisplayText']

def transcribe(path):
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region='westus')
    audio_config = speechsdk.audio.AudioConfig(filename="_dir/tmp.wav")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        return ""

def process(path):
    open('_dir/azure_lock', 'a').close()
    result = transcribe(path)
    print('TRANSCRIPTION: ', result)
    os.remove('_dir/azure_lock')

while True:
    time.sleep(0.1)
    if 'check' in os.listdir('_dir') and os.path.isfile('_dir/gpio_on'):
        process('_dir/tmp.wav')
        os.remove('_dir/check')
        os.remove('_dir/tmp.wav')
