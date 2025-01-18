import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import csv


def preprocess_and_load(path, match_type):
    cat_1_columns = [
        "boundaries",
        "sixes",
        "fifties",
        "hundreds",
        "ducks",
        "thirty_run_innings",  # Renamed from "30_run_innings" to match the original
        "caught",
        "run out",
        "direct",  # Split "direct stumped" into "direct" and "stumped"
        "stumped",
        "3+catches",
        "wickets_taken",
        "3wickets_haul",
        "5wickets_haul",
        "maiden_overs",
        "wickets_lbw_bowled",  # Fixed from "wickets_lbw bowled"
    ]

    cat_2_columns = [
        "dot_balls",
        "total_runs",
        "balls_faced",
        "strike_rate",
        "runs_conceded",
        "balls_bowled",
        "economy_rate",
        "dots",
        "bowling_average",
    ]

    cat_1_windows = [10, 30, 50]  # Sparse
    cat_2_windows = [3, 5, 7]  # Dense

    numerical_features = []
    # Define alpha values for EWMA
    ewma_alphas = [0.5, 0.7, 0.9]

    window = 30
    for col in cat_1_columns:
        col_name = f"{col}_hcma_w{window}"
        numerical_features.append(col_name)

    window = 5
    alpha = 0.7  # Process Cat-2 (Dense: EWMA)
    for col in cat_2_columns:
        col_name = f"{col}_ewma_w{window}_alpha{alpha}"
        numerical_features.append(col_name)

    # Step 1: Load the data
    # Assuming the CSV file is named 'combined_player_data.csv' and is in the current directory
    # combined_df = pd.read_csv('combined_player_data.csv', encoding='ISO-8859-1')
    combined_df = pd.read_csv(path)
    # Filter to only T20 match_type
    # combined_df = combined_df[combined_df['match_type'] == 't20']

    # columns_to_calculate = [
    #     "boundaries", "sixes", "dot_balls", "total_runs", "balls_faced",
    #     "strike_rate", "fifties", "hundreds", "ducks", "thirty_run_innings",
    #     "is_out", "caught", "run out", "direct", "stumped", "3+catches",
    #     "runs_conceded", "balls_bowled", "economy_rate", "wickets_taken",
    #     "3wickets_haul", "5wickets"_haul", "dots", "maiden_overs",
    #     "balls_per_wicket", "bowling_average", "number_of_matches",
    #     "wickets_lbw_bowled", "total_overs_bowled"
    # ]
    # ewma_windows = [8, 10, 12, 11]
    # ewma_alphas = [0.5,0.7,0.9]
    # window =2
    # alpha = 1
    # # Define numerical features

    # numerical_features = []
    # for i in columns_to_calculate:
    #     numerical_features.append(f"{i}_ewma_w{ewma_windows[window]}_a{ewma_alphas[alpha]}")
    #     numerical_features.append(f"{i}_historical_avg")
    # # numerical_features2= [
    # 'boundaries_ewma_w5_a0.5','sixes_ewma_w5_a0.5','dot_balls_ewma_w5_a0.5','total_runs_ewma_w5_a0.5'
    # ]
    # print(numerical_features)

    # Select numerical features and necessary columns
    numerical_df = combined_df[
        ["match_id", "player_id", "date", "label"] + numerical_features
    ].copy()

    print("Number of data points: ", len(numerical_df))

    # Convert 'date' to datetime format
    numerical_df["date"] = pd.to_datetime(numerical_df["date"])

    # Sort by player_id and date
    # numerical_df = numerical_df.sort_values(by=['player_id', 'date']).reset_index(drop=True)

    # Convert numerical features to numeric, handling errors
    for col in numerical_features + ["label"]:
        numerical_df[col] = pd.to_numeric(numerical_df[col], errors="coerce")

    # Step i: Handle outliers
    old_outliers = []
    new_outliers = []
    for feature in numerical_features:
        Q1 = numerical_df[feature].quantile(0.25)
        Q3 = numerical_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (numerical_df[feature] < lower_bound) | (
            numerical_df[feature] > upper_bound
        )
        num_outliers = outliers.sum()
        old_outliers.append(num_outliers)
        if (numerical_df[feature] >= 0).all():
            numerical_df.loc[outliers, feature] = np.sqrt(
                numerical_df.loc[outliers, feature]
            )
        else:
            # Shift data to make it non-negative, apply transformation only to outliers
            min_value = numerical_df[feature].min()
            shift = -min_value
            numerical_df.loc[outliers, feature] = np.sqrt(
                numerical_df.loc[outliers, feature] + shift
            )

    for feature in numerical_features:
        Q1 = numerical_df[feature].quantile(0.25)
        Q3 = numerical_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (numerical_df[feature] < lower_bound) | (
            numerical_df[feature] > upper_bound
        )
        num_outliers = outliers.sum()
        new_outliers.append(num_outliers)

    removed_outliers = [
        (old_outliers[i] - new_outliers[i]) if old_outliers[i] != 0 else -1
        for i in range(len(old_outliers))
    ]
    # Step ii: Calculate skewness and normalize
    # skewness_values = {}

    # for feature in numerical_features:
    #     # Calculate skewness
    #     skewness = skew(numerical_df[feature].dropna())
    #     skewness_values[feature] = skewness
    #     # print(f"Feature {feature}: Skewness after root transformation: {skewness}")
    #     # Normalize using log transformation based on skewness
    #     if skewness > 1 or skewness < -1:
    #         # Log transformation
    #         if (numerical_df[feature] > 0).all():
    #             numerical_df[feature] = np.log(numerical_df[feature])
    #         else:
    #             # Shift data to make it positive
    #             min_value = numerical_df[feature].min()
    #             shift = -min_value + 1e-5  # Add a small epsilon to avoid log(0)
    #             numerical_df[feature] = np.log(numerical_df[feature] + shift)

    # Recalculate skewness after normalization
    for feature in numerical_features:
        skewness = skew(numerical_df[feature].dropna())
        # print(f"Feature {feature}: Skewness after normalization: {skewness}")

    # data_std = numerical_df[numerical_features].std()
    # data_mean = numerical_df[numerical_features].mean()
    # data_std.to_csv(f"../data/interim/scaler/data_std_{match_type}.csv")
    # data_mean.to_csv(f"../data/interim/scaler/data_mean_{match_type}.csv")
    # Step iii: Standardize using mean and variance
    scaler = StandardScaler()

    scaler.fit(numerical_df[numerical_features])
    # print("Test:", scaler.transform(numerical_df[numerical_features])[:-1])
    import joblib

    # print(numerical_df.columns)
    pd.set_option("display.max_columns", None)
    print("---------------------------------------------")
    print("Match Type:", match_type)
    print("Mean:", numerical_df[numerical_features].mean())
    print("Std:", numerical_df[numerical_features].std())

    print(
        "Before scaling: \n",
        numerical_df[
            (numerical_df["player_id"] == "0404d43c")
            & (numerical_df["date"] == "2024-10-09")
        ],
    )

    joblib.dump(scaler, f"../data/interim/scaler/scaler_{match_type}.save")
    # scaler_test = joblib.load(f"../data/interim/scaler/scaler_{match_type}.save")
    # print("Test:", scaler_test.transform(numerical_df[numerical_features])[:-1])

    numerical_df[numerical_features] = scaler.transform(
        numerical_df[numerical_features]
    )
    print(
        "After scaling: \n",
        numerical_df[
            (numerical_df["player_id"] == "0404d43c")
            & (numerical_df["date"] == "2024-10-09")
        ],
    )

    # n_pca = 23
    # pca = PCA(n_components=n_pca)  # Reduce to 18 principal components / Total 23
    # principal_components = pca.fit_transform(numerical_df[numerical_features])
    # principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_pca)])

    principal_df = numerical_df[numerical_features]

    # Use the original numerical features instead of PCA components
    principal_df = numerical_df[numerical_features]

    # Combine with other necessary columns
    final_df = pd.concat(
        [numerical_df[["match_id", "player_id", "date", "label"]], principal_df], axis=1
    )
    final_df.to_csv(path.replace("interim", "processed/formatwise"), index=False)


preprocess_and_load("../data/interim/test.csv", "test")
preprocess_and_load("../data/interim/t20.csv", "t20")
preprocess_and_load("../data/interim/odi.csv", "odi")
