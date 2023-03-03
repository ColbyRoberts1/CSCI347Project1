import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statistics import variance as uvar

# Computes the label encoding of arr
# Assumes arr is a two dimensional numpy array of categorical data
# Assumes arr[0] is an attribute vector
def label_encode(arr: np.ndarray) -> np.ndarray:
    encoder = LabelEncoder()
    result = np.zeros(arr.shape, dtype=int)
    # Label encode each column
    for i, col in enumerate(arr):
        result[i] = encoder.fit_transform(col)
    return result

# Computes the multivariate mean of a numpy array
def mean(arr: np.ndarray) -> np.ndarray:
    return arr.sum(axis = 0)/len(arr)

def variance(arr: np.ndarray) -> np.float64:
    total = 0
    for i in range(len(arr)):
        total += (arr[i] - mean(arr))**2
    return total / (len(arr) - 1)

def standardDeviation(arr: np.ndarray):
    x = 1/(len(arr) - 1)
    total = 0
    for i in range(len(arr)):
        total += (arr[i] - mean(arr))**2
    return np.sqrt(x*total)
    
# Computes the sample covariance of two numpy arrays
def sample_covariance(arr1: np.ndarray, arr2: np.ndarray) -> np.float64:
    arr1mean = mean(arr1)
    arr2mean = mean(arr2)
    x = 1/(len(arr1) - 1)
    total = 0
    for i in range(len(arr1)):
        total += (arr1[i] - arr1mean) *(arr2[i] - arr2mean)
    return total * x

# Computes the correlation coefficient of two numpy arrays
def correlation(arr1: np.ndarray, arr2: np.ndarray) -> np.float64:
    return sample_covariance(arr1, arr2)/(standardDeviation(arr1) * standardDeviation(arr2))

# Computes the total variance for a two dimensional array
# Assumes arr[0] is an attribute vector
def total_variance(arr: pd.DataFrame) -> np.float64:
    total_variance = 0
    for col in arr.columns:
        print(arr[col])
        total_variance += variance(arr[col].array)
    return total_variance

# Loads the data into a Panda's dataframe, sets column names,
def get_data() -> pd.DataFrame:
    # Load and format data as a dataframe
    df = pd.read_csv(
        "./auto-mpg.csv", 
        names=["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin", "Car Make/Model"], 
    )
    # Correct the datatype of Displacement
    df["Displacement"] = df["Displacement"].astype(float)
    return df

# Loads and prepares the dataset
def get_prepared_data():
    df = get_data()
    # Label encode df
    car_name_arr = np.array([df["Car Make/Model"].array])
    df["Car Make/Model"] = pd.Series(label_encode(car_name_arr)[0], dtype=int)
    # Fill missing attributes
    for col in df.columns:
        if df[col].dtype == str:
            continue
        avg = np.mean(df[col])
        df[col].fillna(value=avg, inplace=True)
    return df

if __name__ == "__main__":
    # Test inputs
    THRESHOLD = 0.01
    arr1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    arr2 = np.array([10, 20, 27, 20, 18, 6], dtype=np.float64)
    categorical = np.array([["A", "C", "B", "A", "B"], ["red", "blue", "red", "green", "purple"]])

    # Test label encoding
    expected_labels = np.array([[0, 2, 1, 0, 1], [3, 0, 3, 1, 2.]], dtype=int)
    labels = label_encode(categorical)
    if (not labels.shape == expected_labels.shape):
        print("Label encoding shape test failed!")
        print(f"    expected shape: {expected_labels.shape}")
        print(f"    actual shape: {labels.shape}")
    elif (not (labels == expected_labels).all()):
        print("Label encoding equality test failed!")
        print(f"    expected: {expected_labels}")
        print(f"    actual: {labels}")
    else:
        print("Label encoding tests passed!")

    # Test sample covariance
    covar = sample_covariance(arr1, arr2)
    if (covar + 3.3 < THRESHOLD):
        print("Sample covariance test passed!")
    else:
        print("Sample covariance test failed!")
        print(f"    covariance calculated: {covar}")

    # Test correlation
    corr = correlation(arr1, arr2)
    if (corr + 0.23 < THRESHOLD):
        print("Correlation test passed!")
    else:
        print("Correlation test failed!")
        print(f"    correlation calculated: {corr}")

    # Test total variance
    mat = pd.DataFrame()
    mat["col1"] = pd.Series(arr1, dtype=np.float64)
    mat["col2"] = pd.Series(arr2, dtype=np.float64)
    expected_variance = uvar(arr1) + uvar(arr2)
    calculated_variance = total_variance(mat)
    if calculated_variance - expected_variance < THRESHOLD:
        print("Total variance test passed!")
    else:
        print("Total variance test failed!")
        print(f"    expected: {expected_variance}")
        print(f"    actual: {calculated_variance}")
