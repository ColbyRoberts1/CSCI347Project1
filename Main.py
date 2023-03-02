import numpy as np
import pandas as pd

#THIS IS JUST TO COMPILE THE DATA FILE INTO A NUMPY ARRAY OF OBJECTS. IF YOU FIND AN EASIER WAY TO DO THIS LET ME KNOW OR JUST CHANGE IT LMAO
def makeArr(file):
    fileLines = file.readlines()
    arr = np.empty((len(fileLines),9), dtype=object)
    lineCounter = 0
    while lineCounter < len(fileLines)-1 :
        lineSplit = fileLines[lineCounter].split()
        wordCounter = 0
        while wordCounter < 9:
            if wordCounter == 8:
                carCompName = ''
                while wordCounter < len(lineSplit):
                    carCompName = carCompName + " " + lineSplit[wordCounter]
                    wordCounter += 1
                carCompName = carCompName.replace('"', '')
                carCompName = carCompName[1:]
                arr[lineCounter, 8] = carCompName
            else:
               arr[lineCounter, wordCounter] = lineSplit[wordCounter]
               wordCounter += 1
        lineCounter += 1
    return arr
                
        
with open('auto-mpg.data', 'r') as file:
    data = makeArr(file)
    print(data)
