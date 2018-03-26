import RPi.GPIO as GPIO
from time import sleep
import os
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(17,GPIO.OUT,initial=0)
GPIO.setup(4,GPIO.IN,pull_up_down=GPIO.PUD_UP)

state = False
prevstate = 0

while True:
    if GPIO.input(4) == 0 and prevstate == 1:
        state = not state
    prevstate = GPIO.input(4)
    if state:
        GPIO.output(17,GPIO.HIGH)
        with open('_dir/gpio_on',"w") as f:
            f.write("")
    else:
        try:
            os.remove('_dir/gpio_on')
        except OSError:
            pass
        GPIO.output(17,GPIO.LOW)
    sleep(0.1)
