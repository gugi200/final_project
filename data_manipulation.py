# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:22:44 2024

@author: micha
"""


#
#   Michael Gugala
#   02/12/2023
#   Image recognition
#   Master 4th year project
#   Univeristy of Bristol
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image

import requests
import random
import shutil
import zipfile
from pathlib import Path
import os




PATH_TRAIN = "extended_train"
PATH_TEST = "extended_test"
SHORT_TRAIN = Path("short_train")
SHORT_TEST = Path("short_test")



##############################################
### ROTATING AND TRANSPOSE & ROTATING DATA ###
##############################################

customDirPath = Path("dataset_pressure_sensor/dataCollection1_sensor")
dirs = os.listdir(customDirPath)

#  Create a dir with processed data
extendedDataPath = Path("extended_custom_dataset")
if extendedDataPath.is_dir():
    print('directory already exists')
else:
    extendedDataPath.mkdir(parents=True, exist_ok=True)
    for dir in dirs:
        path = extendedDataPath / dir
        path.mkdir(parents=True, exist_ok=True)


index = 0
for dir in dirs:
    files = os.listdir(customDirPath / dir)

    for file in files:
        img = Image.open(customDirPath / dir / file)
        imgNp = np.asarray(img)
        imgNp_T = np.transpose(imgNp)

        im = Image.fromarray(imgNp)
        im.save(f"{extendedDataPath}/{dir}/{dir}_{index}.jpg")

        im = Image.fromarray(imgNp_T)
        im.save(f"{extendedDataPath}/{dir}/{dir}_{index+1}.jpg")
        for i in range(3):
            imgNp = np.rot90(imgNp)
            imgNp_T = np.rot90(imgNp_T)

            im = Image.fromarray(imgNp)
            im.save(f"{extendedDataPath}/{dir}/{dir}_{(index) + (2*(i+1))}.jpg")

            im = Image.fromarray(imgNp_T)
            im.save(f"{extendedDataPath}/{dir}/{dir}_{(index) + (2*(i+1)) + 1}.jpg")

        index += 8

print('Data mutiplied succesfully')
l = 0
for dir in dirs:
    l += len(os.listdir(extendedDataPath/dir))
    print(dir, len(os.listdir(extendedDataPath/dir)), len(os.listdir(customDirPath/dir)), len(os.listdir(customDirPath/dir))*8)
print(l)


##############################################
### SPLITTING THE DATA INTO TRAIN AND TEST ###
##############################################
TRAIN_RATIO = 0.75
dirs = os.listdir(extendedDataPath)


#  Create a dir for train and test data
extendedTrain = Path(PATH_TRAIN)
extendedTest = Path(PATH_TEST)
if extendedTrain.is_dir():
    print('directory already exists')
else:
    extendedTrain.mkdir(parents=True, exist_ok=True)
    extendedTest.mkdir(parents=True, exist_ok=True)
    for dir in dirs:
        path = extendedTrain / dir
        path.mkdir(parents=True, exist_ok=True)
    for dir in dirs:
        path = extendedTest / dir
        path.mkdir(parents=True, exist_ok=True)

for dir in dirs:
    files = os.listdir(extendedDataPath / dir)
    length = int(TRAIN_RATIO*len(files))
    random.shuffle(files)

    train_set = files[:length]
    test_set = files[length:]

    for data in train_set:
        shutil.copy(extendedDataPath / dir / data, extendedTrain / dir / data)

    for data in test_set:
        shutil.copy(extendedDataPath / dir / data, extendedTest / dir / data)

print('Data split succesfully')
l = 0
for dir in dirs:
    l += len(os.listdir(extendedTrain/dir))
    print(dir, len(os.listdir(extendedTrain/dir)))
print(l)

l = 0
for dir in dirs:
    l += len(os.listdir(extendedTest/dir))
    print(dir, len(os.listdir(extendedTest/dir)))
print(l)


################################################################
### SPLITTING THE DATA INTO TRAIN AND TEST  - TESTING SUBSET ###
################################################################

TRAIN_LENGTH_PER_CLASS = 100
TEST_LENGTH_PER_CLASS = 25
dirs = os.listdir(extendedDataPath)


#  Create a dir for train and test data

if SHORT_TRAIN.is_dir():
    print('directory already exists')
else:
    SHORT_TRAIN.mkdir(parents=True, exist_ok=True)
    SHORT_TEST.mkdir(parents=True, exist_ok=True)
    for dir in dirs:
        path = SHORT_TRAIN / dir
        path.mkdir(parents=True, exist_ok=True)
    for dir in dirs:
        path = SHORT_TEST / dir
        path.mkdir(parents=True, exist_ok=True)

for dir in dirs:
    files = os.listdir(extendedDataPath / dir)
    random.shuffle(files)

    train_set = files[:TRAIN_LENGTH_PER_CLASS]
    test_set = files[TRAIN_LENGTH_PER_CLASS:TRAIN_LENGTH_PER_CLASS+TEST_LENGTH_PER_CLASS]

    for data in train_set:
        shutil.copy(extendedDataPath / dir / data, SHORT_TRAIN / dir / data)

    for data in test_set:
        shutil.copy(extendedDataPath / dir / data, SHORT_TEST / dir / data)

l = 0
for dir in dirs:
    l += len(os.listdir(SHORT_TRAIN/dir))
    print(dir, len(os.listdir(SHORT_TRAIN/dir)))
print(l)

l = 0
for dir in dirs:
    l += len(os.listdir(SHORT_TEST/dir))
    print(dir, len(os.listdir(SHORT_TEST/dir)))
print(l)
