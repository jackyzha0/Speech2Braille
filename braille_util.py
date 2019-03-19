# !/usr/local/bin/python
'''
Author: github.com/jackyzha0
All code is self-written unless explicitly stated
'''
import sys
import time
import string
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def char2braille(char):
    out = [[0,0],[0,0],[0,0]]
    o = ord(char)-97
    k = o % 10
    if o == -65:
        return [[0,0],[0,0],[0,0]]
    if o == 22:
        return [[0,1],[1,1],[0,1]]
    if o == 23:
        return [[1,1],[0,0],[1,1]]
    if o == 24:
        return [[1,1],[0,1],[1,1]]
    if o == 25:
        return [[1,0],[0,1],[1,1]]
    if k <= 7:
        out[0][0]=1
    if k in [2,3,5,6,8,9]:
        out[0][1]=1
    if k in [1,5,6,7,8,9]:
        out[1][0]=1
    if k in [3,4,6,7,9]:
        out[1][1]=1
    if o>9:
        out[2][0]=1
    if o>19:
        out[2][1]=1
    return out

def seq2braille(inp):
    out = ''.join([i for i in inp if not i.isdigit()])
    translator = str.maketrans('', '', string.punctuation)
    out = (list(out.lower().translate(translator)))
    k_out = []
    for i in out:
        k_out.append(char2braille(i))
    return k_out

def disp(arr,s_len=1):
    dic = {0:16,1:26,2:20,3:19,4:21,5:13}
    GPIO.setup(16,GPIO.OUT, initial=0)
    GPIO.setup(26,GPIO.OUT, initial=0)
    GPIO.setup(20,GPIO.OUT, initial=0)
    GPIO.setup(19,GPIO.OUT, initial=0)
    GPIO.setup(21,GPIO.OUT, initial=0)
    GPIO.setup(13,GPIO.OUT, initial=0)
    for i in arr:
        z = [k for j in i for k in j]
        for j in range(len(z)):
            if z[j] == 1:
                GPIO.output(dic[j],GPIO.HIGH)
        time.sleep(s_len)
        GPIO.output(16,GPIO.LOW)
        GPIO.output(26,GPIO.LOW)
        GPIO.output(20,GPIO.LOW)
        GPIO.output(19,GPIO.LOW)
        GPIO.output(21,GPIO.LOW)
        GPIO.output(13,GPIO.LOW)

if __name__ == "__main__":
    seq = seq2braille('test')
    print(seq)
    disp(seq)
