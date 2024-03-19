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
import cv2

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


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



'''
Two additionl modes of operation:
- moving averag datapoint collection with  variable window
- averaged prediction of singular data point


'''

def createRGB(data):
    scaledData = data/4
    scaledData = scaledData.numpy()
    gray_data = cv2.merge([scaledData, scaledData, scaledData])
    tensor = torch.tensor(gray_data).permute([2, 0, 1])
    return transforms.ToPILImage()(tensor.type(torch.uint8))


def expMapping(alpha):
    def func(data):
        data = np.asarray(data)
        data = np.where(data==0, 1, data)
        mapped = np.exp(- (alpha/data))*255
        mapped = mapped.astype(np.uint8)
        return Image.fromarray(mapped)
    return func
        
def softThresholdMapping(lower, upper):
    def func(data):
        data = np.asarray(data)
        th = np.where(data>upper, 255, data)
        th = np.where(th<lower, 0, th)
        return Image.fromarray(th)
    return func

class SensorMgr():


    def __init__(self, root=None, title='picture', imageCount=0):
        self.dataArray = np.zeros((24, 24))
        self.dataReady = False
        self.model_path = "googlenet_test8.pth"
        self.class_names = ['big_fizzy', 'h_big_bottle','h_bottle', 'hand', 'mug', 'small_fizzy']

        self.title = title
        self.rawData = {}
        self.targets = {}

        # prediction variables
        self.newDataFlag = False
        self.MODE_avergae_prediction = True
        self.resetPredMem = True
        self.predThreadLock = False
        self.predCycles = 0
        self.predThreadLockACK = 0
        
        # animation variables
        self.figure = Figure()
        self.pressureImg = self.figure.add_subplot(1, 1, 1)
        self.pressureImg.imshow(np.zeros((24, 24)), vmin=0, vmax=1024)
        self.pressureImg.axis(False)
        
        self.importModel()
        
        self.serialThread = Thread(target=self.connectArduino)
        self.serialThread.start() 

       # average prediction thread
        self.avePredThread = Thread(target=self.avePred)
        self.avePredThread.start()


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
        self.pressureImg.imshow(data, vmin=0, vmax=25)
        if self.dataReady:
            self.predThreadLock = True
            # print('1 aaaaaaaaaaaaaaaaaaaaa')
            # print('thread lock ack: ', self.predThreadLockACK)
            while(self.predThreadLockACK==0):
                # print('wait for prediction thread to respond')
                pass
            # print('2 aaaaaaaaaaaaaaaaaa')
            if self.MODE_avergae_prediction:
            
                # print("results")

                # print(f"Hard predictio: {self.mostCommonClass_Hard}")
                # print(f"Soft prediction: {self.class_names}")
                # print("self.mostCommonClass_Soft: ", self.cumm_pred_prob)
                prob = np.max(self.cumm_pred_prob)/self.predCycles
                pred_class = self.class_names[np.argmax(self.cumm_pred_prob)]
                if prob<0.8:
                    pred_class = 'unrecognised'

                self.pressureImg.set_title(
                    f"Hard predictio: {self.mostCommonClass_Hard} \n"+
                    f"Soft prediction: {pred_class}  \n"
                    f"(probability of {prob*100}%)\n"+
                    f"Num of cycles: {self.predCycles}")

            else:
                prediction, prob, _ = self.makePrediction(self.dataCum/self.predCycles)  
                self.pressureImg.set_title(f"The placed item is {prediction} \n(probability of {prob*100}%)\ncycles: {self.predCycles}")

            self.resetPredMem = True
            self.predThreadLock = False

        self.canvas.draw() 
        self.root.after(1000, self.update)
            
    
    
    
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
            data[rowNumber-1] = sensorValues
            if rowNumber==23:
                self.dataReady = True
                self.newDataFlag = True
                # print(data)
                print('new data available')
                # self.dataArray = (np.exp(- ( 30/(np.array(data)+1) )  )*1023).astype(int)
                self.dataArray = data
                # print(self.dataArray)





    def takePhoto(self):
        pressureData = self.dataArray
        # cam = VideoCapture(0)
        # result, photo = cam.read()
        # cam.release()
        if 1:
            
            self.ResConf = Toplevel(self.root)
            self.ResConf.geometry('1000x700')


            fig, (ax2) = plt.subplots(1, 1, figsize=(10, 7))
            # im1 = ax1.imshow(photo) # later use a.set_data(new_data)
            im2 = ax2.imshow(pressureData) # later use a.set_data(new_data)

            ax2.axis(False)
            self.ResConf.columnconfigure(0, weight=1)
            self.ResConf.columnconfigure(1, weight=1)
            self.ResConf.columnconfigure(2, weight=1)
            self.ResConf.columnconfigure(3, weight=1)
            self.ResConf.columnconfigure(4, weight=1)
            self.ResConf.columnconfigure(5, weight=1)
            self.ResConf.columnconfigure(6, weight=1)
            self.ResConf.columnconfigure(7, weight=1)
            self.ResConf.columnconfigure(8, weight=1)

            buttonSave = Button(self.ResConf, text='Save', command= lambda : self.savePhoto(pressureData))

            buttonSave_class1 = Button(self.ResConf, text=f'Save {self.class_names[0]}', command= lambda : self.savePhoto(pressureData, 1))
            buttonSave_class2 = Button(self.ResConf, text=f'Save {self.class_names[1]}', command= lambda : self.savePhoto(pressureData, 2))
            buttonSave_class3 = Button(self.ResConf, text=f'Save {self.class_names[2]}', command= lambda : self.savePhoto(pressureData, 3))
            buttonSave_class4 = Button(self.ResConf, text=f'Save {self.class_names[3]}', command= lambda : self.savePhoto(pressureData, 4))
            buttonSave_class5 = Button(self.ResConf, text=f'Save {self.class_names[4]}', command= lambda : self.savePhoto(pressureData, 5))
            buttonSave_class6 = Button(self.ResConf, text=f'Save {self.class_names[5]}', command= lambda : self.savePhoto(pressureData, 6))
            buttonSave_class7 = Button(self.ResConf, text=f'Save {self.class_names[6]}', command= lambda : self.savePhoto(pressureData, 7))
            buttonSave_class8 = Button(self.ResConf, text=f'Save {self.class_names[7]}', command= lambda : self.savePhoto(pressureData, 8))

            buttonDiscard = Button(self.ResConf, text='Discard', command=self.ResConf.destroy)
            buttonSave_class1.grid(row=0, column=0)
            buttonSave_class2.grid(row=0, column=1)
            buttonSave_class3.grid(row=0, column=2)
            buttonSave_class4.grid(row=0, column=3)
            buttonSave_class5.grid(row=0, column=4)
            buttonSave_class6.grid(row=0, column=5)
            buttonSave_class7.grid(row=0, column=6)
            buttonSave_class8.grid(row=0, column=7)

            buttonDiscard.grid(row=0, column=len(self.class_names))

            canvas = FigureCanvasTkAgg(fig, master=self.ResConf)

            canvas.get_tk_widget().grid(row=1, columnspan=len(self.class_names)+1, column=0)


    def createDir(self, name):

        os.makedirs(name, exist_ok=True)

    def savePhoto(self, pressureData, target):
        class_name = self.class_names[target-1]

        path_sensor = f'{self.title}_sensor/{class_name}/sensor_{self.saveImgCount}.jpg'
        self.createDir(self.title+'_sensor'+'/'+class_name)
        upper_th = np.where(pressureData<256, pressureData, 255)
        imwrite(path_sensor, upper_th)

        self.createDir(f'{self.title}_rawJSON/{class_name}')



        try:
            with open(f"{self.title}_rawJSON/{class_name}/{class_name}.json", 'r') as f:
                oldData = json.load(f)
        except:
            oldData = {}


        rawData = {}
        with open(f"{self.title}_rawJSON/{class_name}/{class_name}.json", "w") as f:
            rawData[f'sensor_{self.saveImgCount}'] = pressureData.tolist()
            oldData.update(rawData)
            json.dump(oldData, f, indent = 4)
        


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



    def importModel(self):
        
        weight = list(torchvision.models.get_model_weights('googlenet'))[-1]
        self.lodaed_model = torch.hub.load('pytorch/vision', 'googlenet', weight)
        self.lodaed_model.fc = nn.Linear(1024, len(self.class_names), bias=True)


        # class_names = ['big_fizzy', 'h_big_bottle','h_bottle', 'hand', 'mug', 'small_fizzy']
        # self.lodaed_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.lodaed_model.fc = nn.Linear(2048, len(class_names))

        # load model
        self.lodaed_model.load_state_dict(torch.load(f=self.model_path))


    def avePred(self):
        # collect the average prediction with singular data for a period 1 sec
        cumm_pred_prob = np.zeros((len(self.class_names),1))
        preds = []
        
        while(1):

            if self.dataReady and self.newDataFlag:

                # print('data ready Flag: ', self.dataReady)
                # print('new data flag: ', self.newDataFlag)
                # print('reset prediction memory flag: ', self.newDataFlag)
                # print('cycle: ', self.predCycles)
                # print('prediction lock flag: ', self.predThreadLockACK)

                if self.resetPredMem:
                    preds = []
                    self.cumm_pred_prob = np.zeros((1, 6))
                    self.dataCumm = np.zeros((24, 24))
                    self.predCycles = 0
                    self.resetPredMem = False

                data = self.dataArray
                
                # average prediction of n datapoints
                if self.MODE_avergae_prediction:
                    pred_class, class_prob, probMatrix = self.makePrediction(data)
                    self.cumm_pred_prob += probMatrix.numpy()
                    preds.append(pred_class)
                    self.mostCommonClass_Hard = max(set(preds), key=preds.count)
                    np.argmax(self.cumm_pred_prob)

                # prediction of average datapoint over n datapoints
                else:
                    self.dataCumm += data
            
                self.newDataFlag = False
            
                self.predCycles += 1

            while(self.predThreadLock):
                self.predThreadLockACK = 1
            
            self.predThreadLockACK = 0
                




    def makePrediction(self, data):

        preprocess = transforms.Compose([
            createRGB,
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        processed_data = preprocess(torch.tensor(data))




        self.lodaed_model.eval()
        with torch.inference_mode():
            y_preds = self.lodaed_model(processed_data.unsqueeze(dim=0))
            pred_target = y_preds.argmax(dim=1).squeeze()
            probMatrix = torch.softmax(y_preds, dim=1).squeeze()
            class_prob = probMatrix[pred_target]
            pred_class = self.class_names[pred_target]


        return pred_class, class_prob, probMatrix





    

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







