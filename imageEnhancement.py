import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import random

from torchvision.datasets import FashionMNIST


class EnhanceImage():

    def __init__(self, fileSource):
        ''''''
        self.jsonData = EnhanceImage.loadJson(fileSource)
        print(self.jsonData)

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
                newValue = np.where(value<ths, 0, 255)
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

        return dataList, minMax
    

    
    def expomential(self, coeffs):
        '''
        f(x) = exp( coeff * x )
        input: [coeff1, coeff2, coeff3]
        output: [dict1, dict2, dict3], [[min1, max1], [min2, max2], [min3, max3]]
        dict{   'filename': image array}
        '''

        dataList = []
        for coeff in coeffs:
            vmin = 0
            vmax = np.exp(coeff * 1023)
            data = {}
            for key, value in self.jsonData.items():

                value = np.array(value)
                newValue = np.exp(coeff * value)     
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
    

    def displayData(self, originalData, dataList, coeffs, title):
        ''''''



        nrows = 2
        ncols = len(coeffs)
        
        dataFromDicts, originalDataFromDicts = self._dataFromDict(dataList, seed=42, originalData=originalData)

        fig = plt.figure(size=(9, 9))

        for index, data in enumerate(dataFromDicts):

            fig.add_subplot(nrows=nrows, ncols=ncols, index=index)
            plt.imshow(data, cmap='gray')
            plt.title(f"{title} - {coeffs[index]}")
            plt.axis(False)

        for index, data in enumerate(originalDataFromDicts):

            fig.add_subplot(nrows=nrows, ncols=ncols, index=ncols+index)
            plt.imshow(data, cmap='gray')
            plt.title(f"Original data")
            plt.axis(False)


    def analiseData(self, dataList, coeffs, title):

        nrows = 2
        ncols = len(coeffs)
        dataFromDicts = self._dataFromDicts(dataList, seed=42)


        fig = plt.figure(size=(9, 9))

        for index, data in enumerate(dataFromDicts):

            # dispaly the data
            fig.add_subplot(nrows=nrows, ncols=ncols, index=index)
            plt.imshow(data, cmap='gray')
            plt.title(f"{title} - {coeffs[index]}")
            plt.axis(False)

        for index, data in enumerate(dataFromDicts):
            
            # histogram of the data
            fig.add_subplot(nrows=nrows, ncols=ncols, index=ncols+index)
            plt.hist(np.array(data).shape(1, -1), bins = 15)
            plt.title(f"Histogram: {title} - {coeffs[index]}")

            





    def _dataFromDicts(self, dataList, seed=None, originalData=None):
        dataLists_len = len(dataList)
        dataLen = len(dataList[0])

        if seed:
            random.seed(seed)
            
        randInt = random.randint(0, dataLen)

        dataSample = []
        originalDataSample = []
        for data in dataList:
            dataSample.append(data.values()[randInt])

            if originalData:
                originalDataSample(originalData.values()[randInt])


        if originalData:
            return dataSample, originalDataSample

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
    fashionData = FashionMNIST('~pytorch/F_MNIST_data', download=True)
    dataSample = fashionData.data[:40].numpy()

    dimmedData = (dataSample/4).astype(int)

    return dimmedData


if __name__=='__main__':
    
    args = sys.argv
    # if len(args)>=2:
    #     files
    
    
    
    
    
    fileSource = 'awfjhawbf'


    enhanceImg = EnhanceImage(fileSource)
