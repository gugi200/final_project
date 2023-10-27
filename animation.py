from pdb import line_prefix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import json
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import serial
from threading import Thread
import logging
from tkinter import (RIGHT, Label, Scrollbar, Frame, Tk, ttk, NORMAL, DISABLED,
Canvas, BOTH,VERTICAL, LEFT, Y, Button, Menu, Scale, messagebox, Toplevel, StringVar, IntVar, NW, TOP)
from cv2 import VideoCapture, imwrite
from PIL import Image, ImageTk 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation








class Plot_graph():
    '''the class creates a live plot that monitors the speed of 
    the ball moving in the maze'''

    def __init__(self, figure):
        
        # create a fig
        self.figure = figure


        self.serialThread = Thread(target=self.connectArduino)
        self.serialThread.start()
        self.dataArray = np.zeros((24, 24))
        self.dataReady = False

        #axis of the grid
        self.pressureImg = self.figure.add_subplot(1, 1, 1)
        self.pressureImg.imshow(np.zeros((24, 24), vmin=0, vmax=1024))
        self.pressureImg.axis(False)
                

    def connectArduino(self):

        '''
        There will be packets sent of 49 bytes each
        [0] - row number
        [1: ] - values for that row, two bytes per value
        [49:53] - 255 0 255 0 end of coluimn sequence

        When all 24 rows are collected data is updated
        '''
        data = np.zeros((24, 24), dtype=int)
        stopByte1 = 255
        stopByte0 = 0

        ser =  serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=60) 
        ser.close()
        ser.open()
        while (ser.is_open):
            line = ser.read_until(
                    expected = stopByte1.to_bytes(1, 'little') + stopByte0.to_bytes(1, 'little') + \
                                    stopByte1.to_bytes(1, 'little') + stopByte0.to_bytes(1, 'little') ,
                    size = 53
                )
            
            if  len(line)!=53:
                logging.error("Wrong byte sequence received")
                continue 

            rowNumber = line[0]
            sensorValues = [int.from_bytes(line[x:x+2], 'little') for x in range(1, 49, 2)]
            
            data[rowNumber] = sensorValues
            if rowNumber==23:
                
                self.dataReady = True
                self.dataArray = data



    def plot(self):
        '''Plots live data on the graphs'''
        while(not self.dataReady):
            continue

        animation = FuncAnimation(self.figure, self.update, frames=200, interval=100)

        

    def update(self, frames):
        '''necessary for animation'''
        data = self.dataArray
 
        self.pressureImg.clear()
        # self.axes_vel.set_xlim(-5, 5)
        # self.axes_vel.set_ylim(-5, 5)
        self.pressureImg.imshow(data, vmin=0, vmax=1024)





