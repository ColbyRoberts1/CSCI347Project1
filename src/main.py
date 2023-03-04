from dataclasses import dataclass
import pandas as pd
from utils import get_prepared_data, sample_covariance, mean, total_variance, variance, covarianceMatrix, standardNormalization, rangeNormalization

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
    print("Answer:", (list(mean(df))))
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
    
    # What is the covariance matrix of the numerical data matrix (where categorical data  have been converted to numerical values)?
    print("Covariance Matrix:")
    print(covarianceMatrix(df))
    print()
    
    print("Standard Normalized Matrix:")
    print(standardNormalization(df))
    print()
    
    print("Range Normalized Matrix:")
    print(rangeNormalization(df))
    print()
    
    df_cov = rangeNormalization(df)
    coord = np.where(df_cov == np.amax(df_cov))
    print("Which range-normalized numerical attributes have the greatest sample covariance?")
    print("Answer:", cars_num.columns[coord[0]], cars_num.columns[coord[1]], '\n')
    
    print("What is their sample covariance?")
    print("Answer:", np.amax(df_cov), '\n')

    # What is the total variance of the data?
    calculated_variance = total_variance(df)
    print("What is the total variance of the data?")
    print("Answer:", calculated_variance)
    print()

    # What is the total variance of the data, restricted to the five features 
    # that have the greatest sample variance?
    top_variance_df = get_top_variance_cols(df, 5)
    calculated_variance_five = total_variance(top_variance_df)
    print("What is the total variance of the data, restricted to the five features that have the greatest sample variance?")
    print("Answer:", calculated_variance_five)
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

def get_top_variance_cols(df: pd.DataFrame, n: int) -> pd.DataFrame:
    variances = []
    for col in df.columns:
        vari = variance(df[col].array)
        variances.append((vari, col))
    sorted_variances = sorted(variances, key=lambda x: x[0], reverse=True)[:n]
    top_cols = list(map(lambda x: x[1], sorted_variances))
    return df.filter(top_cols, axis="columns")

if __name__ == '__main__':
    report()
