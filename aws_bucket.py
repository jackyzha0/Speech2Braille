import os
import time
import boto3

AWS_s3_client = boto3.client('s3')

def process(path):
    open('_dir/aws_lock', 'a').close()
    AWS_s3_client.upload_file(path, 'speech2wavbucket', path)
    os.remove('_dir/aws_lock')

while True:
    time.sleep(0.1)
    if 'check' in os.listdir('_dir') and os.path.isfile('_dir/gpio_on'):
        process('_dir/tmp.wav')
        os.remove('_dir/check')
        os.remove('_dir/tmp.wav')
