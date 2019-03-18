
import RPi.GPIO as GPIO
from time import sleep
import os
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(17,GPIO.OUT,initial=0)
GPIO.setup(4,GPIO.IN,pull_up_down=GPIO.PUD_UP)

state = False
prevstate = 0
state_changed = 0

green_c="\033[92m"
red_c="\033[91m"
yellow_c="\033[93m"
end_c="\033[0m"
os.system("tput cup 9 0; tput ed")
print("%sIDLE%s mode, press %sButton%s to start..." %(green_c, end_c, yellow_c, end_c))
while True:
    if GPIO.input(4) == 0 and prevstate == 1:
        state = not state
        state_changed = 1
    else:
        state_changed = 0
    prevstate = GPIO.input(4)
    if not state_changed:
        sleep(0.1)
        continue

    if state:
        os.system("tput cup 9 0; tput ed")
        print("%sRECORDING%s mode, press %sButton%s to stop ..." %(red_c, end_c, yellow_c, end_c))
        GPIO.output(17,GPIO.HIGH)
        with open('_dir/gpio_on',"w") as f:
            f.write("")
    else:
        os.system("tput cup 9 0")
        print("%sIDLE%s mode, press %sButton%s to start ..." %(green_c, end_c, yellow_c, end_c))
        try:
            os.remove('_dir/gpio_on')
        except OSError:
            pass
        GPIO.output(17,GPIO.LOW)
    sleep(0.1)
