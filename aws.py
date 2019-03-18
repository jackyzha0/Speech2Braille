def process(path):
    print(path)

while True:
    time.sleep(0.1)
    if 'check' in os.listdir('_dir') and os.path.isfile('_dir/gpio_on'):
        process('_dir/tmp.wav')
        os.remove('_dir/check')
        os.remove('_dir/tmp.wav')
