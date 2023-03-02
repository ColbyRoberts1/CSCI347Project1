import numpy as np
from sklearn.preprocessing import LabelEncoder

# Computes the label encoding of arr
# Assumes arr is a two dimensional numpy array of categorical data
# Assumes arr[0] is a column
def label_encode(arr: np.ndarray) -> np.ndarray:
    encoder = LabelEncoder()
    result = np.zeros(arr.shape, dtype=int)
    # Label encode each column
    for i, col in enumerate(arr):
        print(i, col)
        result[i] = encoder.fit_transform(col)
    return result

# Computes the sample covariance of two numpy arrays
def sample_covariance(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.cov(arr1, arr2)[0][1]

# Computes the correlation coefficient of two numpy arrays
def correlation(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.corrcoef(arr1, arr2)[0][1]

if __name__ == "__main__":
    # Test inputs
    THRESHOLD = 0.01
    arr1 = np.array([1, 2, 3, 4, 5, 6])
    arr2 = np.array([10, 20, 27, 20, 18, 6])
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

    # Test correlation coefficient
    corr = correlation(arr1, arr2)
    if (corr + 0.23 < THRESHOLD):
        print("Correlation coefficient test passed!")
    else:
        print("Correlation coefficient test failed!")
        print(f"    correlation calculated: {corr}")
