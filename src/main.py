from dataclasses import dataclass

import numpy as np
import pandas as pd
from utils import get_data, get_prepared_data, sample_covariance, mean

@dataclass
class Comparison:
    is_greater: bool
    is_equal: bool
    threshold: float

# Report is the function that prints the answer to each question
def report():
    # Dataset initialization
    df = get_prepared_data()
    
    # What is the multivariate mean of the numerical data matrix (where categorical data  have been converted to numerical values)?
    print("What is the multivariate mean of the dataset?")
    print("Answer:") 
    print(mean(df))
    print()

    # How many pairs of features have correlation greater than or equal to 0.5?
    matches_found = calc_matching_pairs_for_threshold(df, Comparison(True, True, 0.5))
    print("How many pairs of features have correlation greater than or equal to 0.5?")
    print("Answer:", matches_found)
    print()

    # How many pairs of features have negative sample covariance?
    matches_found = calc_matching_pairs_for_threshold(df, Comparison(False, False, 0))
    print("How many pairs of features have negative sample covariance?")
    print("Answer:", matches_found)
    print()

def calc_matching_pairs_for_threshold(df: pd.DataFrame, comparison: Comparison) -> int:
    matches = 0
    # Use each feature set once
    for i, first_col in enumerate(df.columns):
        arr1 = df[first_col]    
        # Only compare to unseen feature pairs 
        # We don't want to double count covar(a,b) and covar(b,a) as two matches
        for second_col in df.columns[i+1:]:
            arr2 = df[second_col]
            # Increment tracking var if the pair's corr coeff is GTE comparison.threshold
            if comparison.is_greater and comparison.is_equal:
                if sample_covariance(arr1, arr2) >= comparison.threshold:
                    matches += 1
            # Increment tracking var if the pair's corr coeff is LT comparison.threshold
            if not comparison.is_greater and not comparison.is_equal:
                if sample_covariance(arr1, arr2) < comparison.threshold:
                    matches += 1
    return matches

if __name__ == '__main__':
    report()
