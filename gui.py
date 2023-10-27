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
from animation import Plot_graph
from tkVideoPlayer import TkinterVideo
from matplotlib.figure import Figure


from matplotlib.animation import FuncAnimation, PillowWriter




# pip install tkvideoplayer

class Interface():


    def __init__(self, root, figure):
        '''Initialization function'''


        self.root = root
        self.root.geometry('1000x1000')
        self.root.title('solve the maze')
        self.figure = figure




    def createCanvas(self, mainFrame):
        '''creates canvases'''

        canvas = Canvas(mainFrame)
        canvas.pack(side=LEFT, fill=BOTH, expand=1)
        return canvas 


    def createMainFrame(self):
        '''creates the main frame omto everything is attached to'''
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=8)
        self.createMenuFrame(self.root)
        dataFrame = self.createDataFrame(self.root)
        animFrame = self.createAnimationFrame()
        videoFrame = self.createVideoFrame()
        dataFrame.grid(row=0, column=0, sticky='nw')
        animFrame.grid(row=0, column=1, sticky='ne')
        videoFrame.grid(row=1, column=1, sticky='ne')


        # mainFrame = Frame(self.root)
        # main
        # mainFrame.pack(fill=BOTH, expand=1)
        # self.mainFrame = mainFrame

    def createAnimationFrame(self): 
        frame = Frame(self.root, background='#ffffff', highlightbackground='#000000', 
            highlightthickness=2, width=5000, height=5000, padx=3, pady=3)
        canvas = FigureCanvasTkAgg(self.figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        return frame
        


    def createMenuFrame(self, canvas):
        ''' creates the menu bar'''

        #creating menu
        menuFrame = Frame(canvas)
        # canvas.create_window((0, 0), window=menuFrame, anchor='nw')
        mainMenu = Menu(menuFrame)

        # adding dropbar in view
        viewMenu = Menu(mainMenu, tearoff=False)
        viewMenu.add_command(label='Two columns', command=self.addLabel)
        viewMenu.add_command(label='One column', command=self.addLabel)
        mainMenu.add_cascade(label='View', menu=viewMenu)

        # mainMenu.add_command(label='Load', command=self.load)
        mainMenu.add_command(label='Refresh', command=self.addLabel)
        mainMenu.add_command(label='Clear data', command=self.addLabel)
        mainMenu.add_command(label='Exit', command=self.root.destroy)    
        
        self.root.config(menu=mainMenu)


    def createDataFrame(self, canvas):
        '''frame for displaying parameters'''

        frame = Frame(self.root, background='#ffffff', highlightbackground='#000000', 
            highlightthickness=2, padx=5, pady=5)
        # canvas.create_window((0, 0), window=frame, anchor='nw')

        self.addLabel(frame)

        return frame

    def createVideoFrame(self):
        frame = Frame(self.root, background='#ffffff', highlightbackground='#000000', 
            highlightthickness=2, padx=5, pady=5)
        self.videoplayer = TkinterVideo(master=frame, scaled=True)
        self.videoplayer.load(r"rickrolled.mp4")
        # self.videoplayer.play()
        # self.videoplayer.pack(side=TOP, fill=BOTH, expand=True)
        return frame


    def createAnimateFrame(self, canvas):
        '''frame for displaying parameters'''

        # AnimateFrame = Frame(canvas, background='#ffffff', highlightbackground='#000000', padx=0.1, pady=0.1)
        # canvas.create_window((0, 500), window=AnimateFrame, anchor='nw')
        # self.AnimateFrame = AnimateFrame

    def run(self):
        dataFrame = self.DataFrame
        self.addAnimation()
        self.addLabel(dataFrame=dataFrame)

    
    def addAnimation(self):
        '''adding animation to the AnimateFrame'''
        canvas = FigureCanvasTkAgg(self.figure, self.mainFrame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def play(self):
        result, photo = self.cam.read()
        if result:
            self.photo=photo
            self.photoPressure = self.dataArray
            ResConf = Toplevel(self.root)
            ResConf.geometry('500x500')


            fig, (ax1, ax2) = plt.subplots(1, 2)
            im1 = ax1.imshow(photo) # later use a.set_data(new_data)
            im2 = ax1.imshow(self.dataArray) # later use a.set_data(new_data)

            ax1.axis(False)
            ax2.axis(False)

            canvas = FigureCanvasTkAgg(fig, master=ResConf)

            canvas.get_tk_widget().pack()

            buttonSave = Button(ResConf, text='Save', command=self.savePhoto)
            buttonDiscard = Button(ResConf, text='Discard', command=ResConf.destroy)
            buttonSave.pack()
            buttonDiscard.pack()

    
    def addLabel(self, dataFrame):
        '''adding a label'''

        dataFrame.columnconfigure(0, weight=1)
        dataFrame.columnconfigure(1, weight=4)
        button = Button(dataFrame, text='test test test', background='#ffffff', command=self.tackePhoto)
        button.grid(column=1, row=1, padx=0.1, pady=0.1)
        label_vel = Label(dataFrame, text='current velocity', background='#ffffff')
        label_vel.grid(column=0, row=2, padx=0.1, pady=0.1)
        button2 = Button(dataFrame, text='test test test', background='#ffffff')
        button2.grid(column=1, row=3, padx=0.1, pady=0.1)
        # for widget in self.DataFrame.winfo_children():
        #     print(widget)
        # for widget in self.AnimateFrame.winfo_children():
        #     print(widget)


    def getData(self):
        '''reads the data from somwehere. Either a json file or an external file or program'''
        




def main():
    figure = Figure()
    root = Tk()
    gui = Interface(root, figure)
    mainFrame = gui.createMainFrame()
    # mainCanvas = gui.createCanvas(mainFrame)
    # gui.createMenuFrame(mainCanvas)
    # gui.createDataFrame(mainCanvas)
    # gui.createAnimateFrame(mainCanvas)
    # gui.run()
    animation = Plot_graph(figure)
    animation = FuncAnimation(figure, animation.update, frames=200, interval=1000)
    root.mainloop()


if __name__ == '__main__':
    main()