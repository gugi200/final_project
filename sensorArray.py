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
from matplotlib.figure import Figure
import json
import os
import pathlib
from pathlib import Path
import sys
import time

'''
requirements:
numpy
tk
pyserial
matplotlib
sudo apt-get install python3-tk
Pillow
sudo apt-get install python3-pil python3-pil.imagetk
'''



class SensorMgr():


    def __init__(self, root=None, title='picture', imageCount=0):
        self.dataArray = np.zeros((24, 24))
        self.dataReady = False
        self.title = title
        self.rawData = {}
        self.targets = {}
        
        # animation variables
        self.figure = Figure()
        self.pressureImg = self.figure.add_subplot(1, 1, 1)
        self.pressureImg.imshow(np.zeros((24, 24)), vmin=0, vmax=1024)
        self.pressureImg.axis(False)
        
        self.serialThread = Thread(target=self.connectArduino)
        self.serialThread.start() 

        self.saveImgCount = imageCount   
        if root:
            self.root = root
            self.createAnimationFrame()
            self.create_tkFrame()  
            self.update()
            # self.startAnimation()




    def startAnimation(self):
        while(not self.dataReady):
            continue
    
        anim = animation.FuncAnimation(self.figure, self.update, frames=200, interval=100)


    def update(self):
        '''necessary for animation'''
        data = self.dataArray
 
        self.pressureImg.clear()
        self.pressureImg.imshow(data, vmin=0, vmax=1024)  
        self.canvas.draw() 
        self.root.after(100, self.update)
    
    
    
    def displaySensorImage(self):
        ''''''
               
            

    def connectArduino(self):

        '''
        There will be packets sent of 49 bytes each
        [0] - row number
        [1: ] - values for that row, two bytes per value
        [49:53] - 255 0 255 0 end of coluimn sequence

        When all 24 rows are collected data is updated
        '''


        #### TESTS ####
        # i = 0
        # while(1):
        #     if i==12:
        #         i=0
        #     time.sleep(1)
        #     self.dataReady = True

        #     self.dataArray = np.random.randint(0, 1023, (24, 24))
        #     self.dataArray[i:i+7, i:i+7] = 0
        #     i += 1
        #####


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





    def takePhoto(self):
        pressureData = self.dataArray
        cam = VideoCapture(0)
        result, photo = cam.read()
        cam.release()
        if result:
            
            self.ResConf = Toplevel(self.root)
            self.ResConf.geometry('1000x700')


            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
            im1 = ax1.imshow(photo) # later use a.set_data(new_data)
            im2 = ax2.imshow(pressureData) # later use a.set_data(new_data)

            ax1.axis(False)
            ax2.axis(False)
            self.ResConf.columnconfigure(0, weight=1)
            self.ResConf.columnconfigure(1, weight=1)
            self.ResConf.columnconfigure(2, weight=1)
            self.ResConf.columnconfigure(3, weight=1)
            self.ResConf.columnconfigure(4, weight=1)
            self.ResConf.columnconfigure(5, weight=1)

            buttonSave = Button(self.ResConf, text='Save', command= lambda : self.savePhoto(pressureData, photo))

            buttonSave_class1 = Button(self.ResConf, text='Save target 1', command= lambda : self.savePhoto(pressureData, photo, 1))
            buttonSave_class2 = Button(self.ResConf, text='Save target 2', command= lambda : self.savePhoto(pressureData, photo, 2))
            buttonSave_class3 = Button(self.ResConf, text='Save target 3', command= lambda : self.savePhoto(pressureData, photo, 3))
            buttonSave_class4 = Button(self.ResConf, text='Save target 4', command= lambda : self.savePhoto(pressureData, photo, 4))
            buttonSave_class5 = Button(self.ResConf, text='Save target 5', command= lambda : self.savePhoto(pressureData, photo, 5))

            buttonDiscard = Button(self.ResConf, text='Discard', command=self.ResConf.destroy)
            buttonSave_class1.grid(row=0, column=0)
            buttonSave_class2.grid(row=0, column=1)
            buttonSave_class3.grid(row=0, column=2)
            buttonSave_class4.grid(row=0, column=3)
            buttonSave_class5.grid(row=0, column=4)

            buttonDiscard.grid(row=0, column=5)

            canvas = FigureCanvasTkAgg(fig, master=self.ResConf)

            canvas.get_tk_widget().grid(row=1, columnspan=6, column=0)


    def createDir(self, name):

        os.makedirs(name, exist_ok=True)

    def savePhoto(self, pressureData, photo, target):

        self.createDir(self.title+'_cam')
        imwrite(self.title+'_cam'+f'/{self.title}_cam_{self.saveImgCount}.jpg', np.interp(photo, [0,1023],[0,255]).astype(int))
        self.createDir(self.title+'_sensor')
        imwrite(self.title+'_sensor'+f'/{self.title}_sensor_{self.saveImgCount}.jpg', np.interp(pressureData, [0,1023],[0,255]).astype(int))

        self.createDir(self.title+'_rawJSON')
        self.createDir(self.title+'_target')


        try:
            with open(self.title+'_rawJSON'+"/rawData.json", 'r') as f:
                oldData = json.load(f)
        except:
            oldData = {}
        try:
            with open(self.title+'_target'+"/target.json", 'r') as f:
                oldTargets = json.load(f)
        except:
            oldTargets = {}


        with open(self.title+'_rawJSON'+"/rawData.json", "w") as f:
            self.rawData[f'{self.title}_sensor_{self.saveImgCount}'] = pressureData.tolist()
            oldData.update(self.rawData)
            json.dump(oldData, f, indent = 4)
        
        with open(self.title+'_target'+"/target.json", "w") as f:
            self.targets[f'{self.title}_sensor_{self.saveImgCount}'] = target
            oldTargets.update(self.targets)
            json.dump(oldTargets, f, indent = 4)

        self.saveImgCount += 1
        self.ResConf.destroy()
       




    def create_tkFrame(self):
        ''''''
        self.root.geometry('1000x700')
        self.root.title('Data collection - pressure sensor array')
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=4)
        animFrame = self.createAnimationFrame()
        buttonFrame = self.createButtonFrame()
        buttonFrame.grid(row=0, column=0, sticky='nw')
        animFrame.grid(row=0, column=1, sticky='ne')





    def createAnimationFrame(self): 
        frame = Frame(self.root, background='#ffffff', highlightbackground='#000000', 
            highlightthickness=2, width=400, height=500, padx=3, pady=3)
        self.canvas = FigureCanvasTkAgg(self.figure, frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        return frame
    


    def createButtonFrame(self):
        frame = Frame(self.root, background='#ffffff', highlightbackground='#000000', 
            highlightthickness=2, width=100, height=500, padx=3, pady=3)

        button = Button(frame, text='Take photo', command=self.takePhoto)
        button.pack()
        return frame


    def displayPressureImg(self):
        captured_image = Image.fromarray(self.dataArray)
        photo_image = ImageTk.PhotoImage(image=captured_image) 
        self.label_widget.imgtk = photo_image
        self.label_widget.configure(image=photo_image)
        self.label_widget.after(1, self.displayPressureImg)





    

if __name__=='__main__':
    args = sys.argv
    if len(args)==1:
        title='test'
    elif len(args)==2:
        title = str(args[1])
    else:
        title = str(args[1])
        imageCount = int(args[2])

    root = Tk()
    sensor = SensorMgr(root=root, title=title, imageCount=imageCount)
    # sensor.displaySensorImage()
    root.mainloop()







