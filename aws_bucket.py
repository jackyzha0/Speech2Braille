import os
import time
import boto3

AWS_s3_client = boto3.client('s3')
AWS_transcribe_client = boto3.client('transcribe')
BUCKETPATH = 'https://s3.amazonaws.com/s2bwavbucket/_dir/tmp.wav'

def transcribe(JOB_URL):
    try:
        response = AWS_transcribe_client.delete_transcription_job(
            TranscriptionJobName='speech2txt'
        )
        print('Delete Previous Job Status: ', response)
    except:
        print('No previous job found!')
    AWS_transcribe_client.start_transcription_job(
        TranscriptionJobName='speech2txt',
        Media={'MediaFileUri': JOB_URL},
        MediaFormat='wav',
        LanguageCode='en-US'
    )

    while True:
        status = AWS_transcribe_client.get_transcription_job(TranscriptionJobName='speech2txt')
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Not ready yet...")
        time.sleep(1)
    return(status)

def process(path):
    open('_dir/aws_lock', 'a').close()
    AWS_s3_client.upload_file(path, 's2bwavbucket', path)
    print('NEWFILE')
    result = transcribe(BUCKETPATH)
    print('TRANSCRIPTION: ', result)
    os.remove('_dir/aws_lock')

while True:
    time.sleep(0.1)
    if 'check' in os.listdir('_dir') and os.path.isfile('_dir/gpio_on'):
        process('_dir/tmp.wav')
        os.remove('_dir/check')
        os.remove('_dir/tmp.wav')
