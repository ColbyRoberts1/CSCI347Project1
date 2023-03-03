import numpy as np
#import pandas as pd

#THIS IS JUST TO COMPILE THE DATA FILE INTO A NUMPY ARRAY OF OBJECTS. IF YOU FIND AN EASIER WAY TO DO THIS LET ME KNOW OR JUST CHANGE IT LMAO
def makeArr(file):
    fileLines = file.readlines()
    arr = np.empty((len(fileLines)-1,9))
    lineCounter = 0
    carModel = 0
    while lineCounter < len(fileLines)-1 :
        lineSplit = fileLines[lineCounter].split()
        wordCounter = 0
        while wordCounter < 9:
            #THIS IS FOR THE PURPOSE OF LABEL-ENCODING THE CAR MODEL NAME. BECAUSE THEY ARE ALL DIFFERENT IT WILL JUST BE ITERATING BY ONE EACH TIME
            if wordCounter == 8:
                arr[lineCounter, wordCounter] = carModel
                carModel += 1
                wordCounter = 9
            else:
                if lineSplit[wordCounter] == '?':
                    arr[lineCounter, wordCounter] = 0
                else: 
                    arr[lineCounter, wordCounter] = lineSplit[wordCounter]
                wordCounter += 1
        lineCounter += 1
    return arr          
        
def findMean(arr):
    mean = arr.sum(axis = 0)/arr.shape[0]
    return mean
    

def rangeNorm(arr):
    minimum = arr.min()
    maximum = arr.max()
    return (arr - minimum)/(maximum - minimum)

def standardDeviation(arr):
    x = 1/(len(arr) - 1)
    total = 0
    for i in range(len(arr)):
        total += (arr[i] - (arr))**2
    return np.sqrt(x*total)

def vectorMean(arr):
    return arr.sum(axis=0) / arr.shape[0]

def standNorm(arr):
    vector_mean = vectorMean(arr)
    return (arr - vector_mean)/standardDeviation(arr)

with open('auto-mpg.data', 'r') as file:
    data = makeArr(file)
    print(data)
    print(rangeNorm(data))
    print(standNorm(data))
    print(findMean(data))


