import os 
from pathlib import Path
import json
import sys


class RemovaData():


    def __init__(self, mode, files=None):
        self.camDirPath = 'dataCollection1_sensor'
        self.jsonDirPath = 'dataCollection1_rawJSON'
        # self.camDirPath = 'test_sensor'
        # self.jsonDirPath = 'test_rawJSON'
        self.files = files

        if mode=='rm':
            self.remove()

        if mode=='hv':
            self.findValue()


    def removeJSON(self, datapoint):
        dirs = os.listdir(self.jsonDirPath+'/')
        for dir in dirs:
            jsonFile = os.listdir(self.jsonDirPath+'/'+dir+'/')[0]
            print(jsonFile)
            with open(self.jsonDirPath+'/'+dir+'/'+jsonFile, 'r') as f:
                jsonData = json.load(f)
            if datapoint in jsonData:
                del jsonData[datapoint]
                with open(self.jsonDirPath+'/'+dir+'/'+jsonFile, 'w') as f:
                    json.dump(jsonData, f, indent = 4)
                print(f'Json Data point:  {datapoint} removed')

                return
        print('No file with that name found')


    def removeSensor(self, datapoint):
        dirs = os.listdir(self.jsonDirPath+'/')
        for dir in dirs:
            files = os.listdir(self.camDirPath+'/'+dir+'/')
            if (datapoint+'.jpg') in files:
                os.remove(self.camDirPath+'/'+dir+'/'+datapoint+'.jpg')
                print(f'Sensor Data point:  {datapoint} removed')
                return
        print('No file with that name found')




    def remove(self):
        for dataPoint in self.files:
            self.removeSensor(dataPoint)
            #self.removeJSON(dataPoint)

        ''''''

    def findValue(self):
        dirs = os.listdir(self.camDirPath+'/')
        highestV = 0
        highestFile = ''
        class_name = ''
        print(dirs)
        for dir in dirs:

            # get all files in a folder
            files = os.listdir(self.camDirPath+'/'+dir+'/')
            print(f"dir: '{dir}', count: {len(files)}")
            for file in files:
                fileName = int(file[7:-4])
                if fileName>highestV:
                    highestV = fileName
                    highestFile = file
                    highestDir = dir

        print('Highest value: ', highestV)
        print('File: ', highestFile)
        print('Dir: ', highestDir)

    



if __name__=='__main__':

    args = sys.argv
    print(args)
    if args[1]=='rm':
        files = args[2:]
        RemovaData('rm', files)

    elif args[1]=='hv':
        RemovaData('hv')
    else:
        print('No files have been specified')
