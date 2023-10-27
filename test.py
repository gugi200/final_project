import numpy as np
import serial
from threading import Thread




ser =  serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=60) 
ser.close()
ser.open()
while(True):
    x = ser.read(49)
    print(list(x))




