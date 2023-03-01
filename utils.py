import numpy as np

def sample_covariance(arr1: np.ndarray, arr2: np.ndarray):
    return np.cov(arr1, arr2)[0][1]

def correlation(arr1: np.ndarray, arr2: np.ndarray):
    return np.corrcoef(arr1, arr2)[0][1]

if __name__ == "__main__":
    THRESHOLD = 0.01
    arr1 = np.array([1, 2, 3, 4, 5, 6])
    arr2 = np.array([10, 20, 27, 20, 18, 6])
    # Test sample covariance
    assert(sample_covariance(arr1, arr2) + 3.3 < THRESHOLD)
    print(correlation(arr1, arr2))
