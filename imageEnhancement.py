import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from PIL import Image
import pandas as pd



class EnhanceImage():

    def __init__(self, fileSource):
        ''''''
        # self.jsonData = EnhanceImage.loadJson(fileSource)
        self.jsonData = fileSource

        # mappedData = np.interp(newValue, (min, max), (0, 255)).astype(int)

    @staticmethod
    def loadJson(fileSource: str) -> dict:
        
        return np.random.randint(0, 255, size=(24, 24))
        
        with open(fileSource, "r") as f:
            jsonData = json.load(f)
        
        return jsonData
    
    def threshold(self, ths: list) -> list: 
        '''
        f(x) = 0 if x<th; 255 if >= th
        input:  [th1, th2, th3]
        output: [dict1, dict2, dict3]
        dict{   'filename': image array}
        '''

        thresholdDataList= []
        for th in ths:
            thresholdData = {}
            for key, value in self.jsonData.items():

                value = np.array(value)
                newValue = np.where(value<th, 0, 255)
                thresholdData[key] = newValue
            
            thresholdDataList.append(thresholdData)

        return thresholdDataList

        

    def minMaxLimits(self, minMax):
        '''
        f(x) = 0 if x<min = min;  x if min<= x <= max; max if x>max
        input: [[min1, max1] [min2, max2] [min3, max3]]
        output: [dict1, dict2, dict3]
        dict{   'filename': image array}
        '''

        dataList = []
        for limit in minMax:
            min = limit[0]
            max = limit[1]

            data = {}
            for key, value in self.jsonData.items():

                value = np.array(value)
                newValue = np.where(value<min, min, value)
                newValue = np.where(newValue>max, max, value)
                mappedData = np.interp(newValue, (min, max), (0, 255)).astype(int)
                data[key] = mappedData

            dataList.append(data)

        return dataList
    

    
    def expomential(self, coeffs):
        '''
        f(x) = exp( -(coeff / x) )
        input: [coeff1, coeff2, coeff3]
        output: [dict1, dict2, dict3], [[min1, max1], [min2, max2], [min3, max3]]
        dict{   'filename': image array}
        '''

        dataList = []
        for coeff in coeffs:
            vmin = 0
            vmax = np.exp(-(coeff / 1023))
            data = {}
            for key, value in self.jsonData.items():

                value = np.array(value)
                newValue = np.exp(-(coeff / (value+10)))     
                mappedData = np.interp(newValue, (vmin, vmax), (0, 255)).astype(int)

                data[key] =  mappedData        
            dataList.append(data)
        return  dataList                    


        
    def power(self, coeffs):
        '''
        f(x) = x^(coeff)
        input: [coeff1, coeff2, coeff3]
        output: [dict1, dict2, dict3], [[min1, max1], [min2, max2], [min3, max3]]
        dict{   'filename': image array}
        '''

        dataList = []

        for coeff in coeffs:
            vmin = 0
            vmax = 1023**(coeff)

            data = {}

            for key, value in self.jsonData.items():
                value = np.array(value)
                newValue = value**coeff
                mappedData = np.interp(newValue, (vmin, vmax), (0, 255)).astype(int)

                data[key] = mappedData

            dataList.append(data)
        

        return dataList

    def sinMapping(self, offsets):
        '''
        theta(x) = <0; 90>
        f(x) = tan( theta(x) )
        input:  [coeff1 coeff2 coeff3], [offset1 offset2 offset3], thetaMappingMode={linear}
        output: [dict1, dict2, dict3], [[min1, max1], [min2, max2], [min3, max3]]
        dict{   'filename': image array}
        '''

        dataList = []

        for offset in offsets:
            data = {}
            for key, value in self.jsonData.items():
                
                value = np.array(value) 
                theta = self._thetaMapping(value, offset)
                newValue = 255*np.sin(theta)

                data[key] = newValue.astype(int)

            dataList.append(data)

        return dataList

        
    def logMapping(self, value, coeffs):
        '''
        f(x) = log_coeff(value)
        input: [coeff1, coeff2, coeff3]
        output: [dict1, dict2, dict3], [[min1, max1], [min2, max2], [min3, max3]]
        dict{   'filename': image array}
        '''      

        dataList = []

        for coeff in coeffs:
            data = {}
            vmin = np.log(order=coeff, x=1)
            vmax = np.log(order=coeff, x=1024)
            for key, value in self.jsonData.items():

                value = np.array(value) + 1
                newValue = np.log(order=coeff, x=value) 
                mappedData = np.interp(newValue, (vmin, vmax), (0, 255)).astype(int)

                data[key] = value
            dataList.append(data)

        return data
    

    def displayData(self, originalData, dataList, coeffs, title, verbose=False):
        
        
        nrows = 2

        if verbose:
            nrows *= 2
        ncols = len(coeffs) + 1
        
        dataFromDicts, originalDataFromDicts = self._dataFromDicts(dataList, seed=42, originalData=originalData)

        fig = plt.figure(figsize=(15, 5))

       

        fig.add_subplot(nrows, ncols, 1)
        plt.imshow(originalDataFromDicts, cmap='gray')
        plt.title(f"Original data")
        plt.axis(False)

        for index, data in enumerate(dataFromDicts):
            
            fig.add_subplot(nrows, ncols, index+2)
            plt.imshow(data, cmap='gray')
            plt.title(f"{title} - {coeffs[index]}")
            plt.axis(False)

        if verbose:

            # histogram of the data
            fig.add_subplot(nrows, ncols, ncols+1)
            plt.hist(originalDataFromDicts.ravel(), bins = 10)
            plt.title(f"Histogram: {title} - original")

            for index, data in enumerate(dataFromDicts):
                # histogram of the data
                fig.add_subplot(nrows, ncols, ncols+index+2)
                plt.hist(data.ravel(), bins = 10)
                plt.title(f"Histogram: {title} - {coeffs[index]}")


        fig.tight_layout()

        # plt.show()
        return fig


    def analiseData(self, dataList, coeffs, title):

        nrows = 2
        ncols = len(coeffs)
        dataFromDicts = self._dataFromDicts(dataList, seed=42)


        fig = plt.figure(figsize=(9, 9))

        for index, data in enumerate(dataFromDicts):

            # dispaly the data
            fig.add_subplot(nrows, ncols, index+1)
            plt.imshow(data, cmap='gray')
            plt.title(f"{title} - {coeffs[index]}")
            plt.axis(False)

        for index, data in enumerate(dataFromDicts):
            # histogram of the data
            fig.add_subplot(nrows, ncols, ncols+index+1)
            plt.hist(data.ravel(), bins = 15)
            plt.title(f"Histogram: {title} - {coeffs[index]}")

        return fig

            





    def _dataFromDicts(self, dataList, seed=None, originalData=None):
        dataLen = len(dataList[0])

        if seed:
            random.seed(seed)
            
        randInt = random.randint(0, dataLen-1)
        self.rand = randInt
        dataSample = []
        originalDataSample = []
        for data in dataList:
            valuesDict = np.array(list(data.values()))
            dataSample.append(valuesDict[randInt])

        if originalData:
            valuesDict = np.array(list(originalData.values()))[randInt]



        if originalData:
            return dataSample, valuesDict

        return dataSample





    def _thetaMapping(self, value, offset=0, mode='linear', vmin=0, vmax=1023):
        
        
        if mode=='linear':
            thetaMapped = np.interp(value, (vmin, vmax), (0, np.pi/2))


            return thetaMapped
        
        if mode=='offset':
            newValue = np.where(value>offset, offset, value)
            thetaMapped = np.interp(value, (vmin, offset), (0, np.pi/2))

            return thetaMapped
        

        


             
    





    

            
def importTestDataset():
    from sklearn.datasets import load_digits
    data, target = load_digits(as_frame=True, return_X_y=True)
    rawData = data.values[:40]
    # np.random.seed(42)
    numpyData = rawData.reshape(40, 8, 8)*16 + np.random.randint(0, 100, (8, 8))
    rawData = rawData.reshape(40, 8, 8)
    rawImages = []
    data={}

    for index, datapoint in enumerate(numpyData):
        image = Image.fromarray(datapoint)
        imageRaw = Image.fromarray(rawData[index])

        image = image.resize((24, 24))
        imageRaw = imageRaw.resize((24, 24))

        rawImages.append(np.array(imageRaw).tolist())
        data[f'original_{index}'] = np.array(image).tolist()



    return data, rawImages


def displayRawImage(image):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.show()

if __name__=='__main__':
    
    args = sys.argv
    # if len(args)>=2:
    #     files
    
    
    
    
    
    #fileSource = 'awfjhawbf'

    dictData, rawImages = importTestDataset()

    enhanceImg = EnhanceImage(dictData)

    ### threshold
    ths = [128, 255, 270, 360]
    thresholdData = enhanceImg.threshold(ths)
    figDsiplay = enhanceImg.displayData(dictData, thresholdData, ths, 'threshold', verbose=True)


    ### minMax mapping
    minMax = [[10, 1000], [20, 512], [60, 256], [128, 256]]
    minMaxData = enhanceImg.minMaxLimits(minMax)
    figDsiplay = enhanceImg.displayData(dictData, minMaxData, minMax, 'minMax', verbose=True)



    ### exponential mapping
    coeffs = [20, 60, 90, 128]
    expData = enhanceImg.expomential(coeffs)
    figDsiplay = enhanceImg.displayData(dictData, expData, coeffs, 'exp', verbose=True)



    ### power mapping
    coeffs = [0.1, 0.3, 0.6, 0.8]
    powerData = enhanceImg.power(coeffs)
    figDsiplay = enhanceImg.displayData(dictData, powerData, coeffs, 'power', verbose=True)


    # for debugging on testing dataset
    rand = enhanceImg.rand
    displayRawImage(rawImages[rand])
