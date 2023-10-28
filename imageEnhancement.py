import json
import matplotlib.pyplot as plt
import numpy as np
import sys


class EnhanceImage():

    def __init__(self, fileSource):
        ''''''
        self.jsonData = EnhanceImage.loadJson(fileSource)

        # mappedData = np.interp(newValue, (min, max), (0, 255)).astype(int)

    @staticmethod
    def loadJson(fileSource: str) -> dict:
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

        minMax = []
        thresholdDataList= []
        for th in ths:
            thresholdData = {}
            for key, value in self.jsonData.items():

                value = np.array(value)
                newValue = np.where(value<ths, 0, 1023)
                thresholdData[key] = newValue
            
            thresholdDataList.append(thresholdData)
            minMax.append([0, 1023])

        return thresholdDataList, minMax

        

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

                data[key] = newValue

            dataList.append(data)

        return dataList, minMax
    





    

            






if __name__=='__main__':
    
    args = sys.argv
    # if len(args)>=2:
    #     files
    enhanceImg = EnhanceImage(fileSource)
