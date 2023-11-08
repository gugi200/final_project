
import serial, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from threading import Thread


class SensorMgr():

    def __init__(self):
        self.dataArray = np.zeros((6, 6))
        self.dataReady = False

        

        
        self.serialThread = Thread(target=self.connectArduino)
        self.serialThread.start() 


    def connectArduino(self):

            '''
            There will be packets sent of 13 bytes each
            [0] - row number
            [1: 13] - values for that row, two bytes per value
            [13:17] - 255 0 255 0 end of coluimn sequence

            When all 6 rows are collected data is updated
            '''
            data = np.zeros((6, 6), dtype=int)
            stopByte1 = 255
            stopByte0 = 0

            ser =  serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=60) 
            ser.close()
            ser.open()
            while (ser.is_open):
                line = ser.read_until(
                        expected = stopByte1.to_bytes(1, 'little') + stopByte0.to_bytes(1, 'little') + \
                                        stopByte1.to_bytes(1, 'little') + stopByte0.to_bytes(1, 'little') ,
                        size = 17
                    )
                
                if  len(line)!=17:
                    continue 

                rowNumber = line[0]
                sensorValues = [int.from_bytes(line[x:x+2], 'little') for x in range(1, 13, 2)]
                
                data[rowNumber] = sensorValues
                if rowNumber==5:
                    
                    self.dataReady = True
                    self.dataArray = data
	  

	

		

data = SensorMgr()
fig, ax = plt.subplots()
mat = ax.matshow(data.dataArray,  vmin=0, vmax=1023)

def update(i):
     return mat.set_data(data.dataArray)

ax.autoscale(False)
plt.colorbar(mat)
ani = animation.FuncAnimation(fig, update, frames=200, interval=100)

plt.show()