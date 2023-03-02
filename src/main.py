import numpy as np
import pandas as pd

# Loads the data into a Panda's dataframe, sets column names,
def getData():
    # Load and format data as a dataframe
    df = pd.read_csv(
        "./auto-mpg.csv", 
        names=["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin", "Car Make/Model"], 
    )
    # Correct the datatype of Displacement
    df["Displacement"] = df["Displacement"].astype(float)
    return df

if __name__ == '__main__':
    getData()
