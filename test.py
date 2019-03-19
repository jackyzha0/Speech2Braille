import RPi.GPIO as GPIO
import time, sys
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setup(int(sys.argv[2]), GPIO.OUT, initial=0)

GPIO.output(int(sys.argv[2]), 1 if sys.argv[1] == "1" else 0)
#time.sleep(3)
#GPIO.output(16,0)
