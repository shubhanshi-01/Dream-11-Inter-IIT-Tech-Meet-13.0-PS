import pandas as pd
import numpy as np
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from typing import List, Dict, Any
import joblib
from pathlib import Path
import torch
import torch.nn as nn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def get_model_path(match_type: str) -> str:
    """Get the path to the model file."""
    return f"../../../model_artifacts/Product_UI_{match_type.lower()}_Model.pth"


# Global cache for feature calculations
FEATURE_CACHE = {}
PLAYER_DATA_CACHE = {}

# Constants moved to global scope
FORMAT_MAPPING = {"it20": "t20", "mdm": "test", "odm": "odi"}
CAT_1_COLUMNS = [
    "boundaries",
    "sixes",
    "fifties",
    "hundreds",
    "ducks",
    "thirty_run_innings",
    "caught",
    "run out",
    "direct",
    "stumped",
    "3+catches",
    "wickets_taken",
    "3wickets_haul",
    "5wickets_haul",
    "maiden_overs",
    "wickets_lbw_bowled",
]
CAT_1_WINDOWS = [10, 30, 50]
CAT_2_COLUMNS = [
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
CAT_2_WINDOWS = [3, 5, 7]
EWMA_ALPHAS = [0.5, 0.7, 0.9]


class LargeNet(nn.Module):
    def __init__(self, input_size):
        super(LargeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.fc5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        out = self.fc6(out)  # No activation in output layer (for regression tasks)
        return out


def mape_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    # Ensure inputs have the same shape
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shapes must match. Got y_pred: {y_pred.shape}, y_true: {y_true.shape}"
        )

    # Calculate absolute percentage error
    absolute_percentage_error = (
        torch.abs((y_true - y_pred) / (y_true + epsilon)) * 100.0
    )

    # Return mean over all dimensions
    return torch.mean(absolute_percentage_error)


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def rmse_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def preprocess_player_data(player_id: str) -> pd.DataFrame:
    """Preprocesses and caches player data."""
    if player_id in PLAYER_DATA_CACHE:
        return PLAYER_DATA_CACHE[player_id]

    input_folder = Path("../../../data/processed/playerwise/")
    file_path = input_folder / f"{player_id}.csv"

    try:
        df = pd.read_csv(
            file_path,
            usecols=["date", "match_type", "balls_bowled", "balls_faced"]
            + CAT_1_COLUMNS
            + CAT_2_COLUMNS,
        )

        # Optimize dtypes
        df["match_type"] = df["match_type"].str.lower()
        df["revised_format"] = (
            df["match_type"].map(FORMAT_MAPPING).fillna(df["match_type"])
        )
        df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[D]")

        # Convert numeric columns to float32 for memory efficiency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

        PLAYER_DATA_CACHE[player_id] = df
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def calculate_features_batch(args: tuple) -> tuple:
    """Calculate features for a single player in batch mode."""
    player_id, date, match_type = args
    cache_key = (player_id, date, match_type)

    if cache_key in FEATURE_CACHE:
        return player_id, FEATURE_CACHE[cache_key]

    df = preprocess_player_data(player_id)
    if df.empty:
        features = {"player_role": "new", "score": 0}
        FEATURE_CACHE[cache_key] = features
        return player_id, features

    # Filter data efficiently
    mask = (df["date"] < np.datetime64(date)) & (
        df["revised_format"]
        == FORMAT_MAPPING.get(match_type.lower(), match_type.lower())
    )
    df_filtered = df[mask]

    if df_filtered.empty:
        features = {"player_role": "new", "score": [0]}
        FEATURE_CACHE[cache_key] = features
        return player_id, features

    # Vectorized role determination
    total_matches = len(df_filtered)
    role_metrics = {
        "bowler": (df_filtered["balls_bowled"] > 0).mean() >= 0.25,
        "batsman": (df_filtered["balls_faced"] > 0).mean() >= 0.25,
    }

    player_role = (
        "all-rounder"
        if all(role_metrics.values())
        else (
            "bowler"
            if role_metrics["bowler"]
            else "batsman" if role_metrics["batsman"] else "new"
        )
    )

    # Optimized feature calculation using numpy operations
    features = pd.DataFrame({"player_role": [player_role]})

    # Vectorized calculations for both categories
    for col, windows, is_cat1 in [
        (CAT_1_COLUMNS, CAT_1_WINDOWS, True),
        (CAT_2_COLUMNS, CAT_2_WINDOWS, False),
    ]:
        for c in col:
            if c in df_filtered.columns:
                values = df_filtered[c].values
                if is_cat1:
                    for window in windows:
                        features[f"{c}_hcma_w{window}"] = (
                            pd.Series(values)
                            .rolling(window=window, min_periods=1)
                            .mean()
                            .iloc[-1]
                        )
                else:
                    for window in windows:
                        for alpha in EWMA_ALPHAS:
                            features[f"{c}_ewma_w{window}_alpha{alpha}"] = (
                                pd.Series(values)
                                .ewm(alpha=alpha, adjust=False)
                                .mean()
                                .iloc[-1]
                            )

    # Calculate a preliminary score based on features
    features["score"] = model_inference(features.iloc[0], match_type)
    FEATURE_CACHE[cache_key] = features
    return player_id, features


def get_actual_scores(team_players1, team_players2, match_date, match_type):
    scores = []
    data = pd.read_csv(f"../../../data/processed/formatwise/{match_type}.csv")

    for player in team_players1:
        player_id = player["id"]
        player_data = data[
            (data.player_id == player_id) & (data.date == match_date)
        ].reset_index()
        # print(data[data.date == match_date].shape)
        if player_data.shape[0] != 0 and not np.isnan(player_data.label[0]):
            scores.append((player_data.label[0], player_data.player_id[0]))

    for player in team_players2:
        player_id = player["id"]
        player_data = data[
            (data.player_id == player_id) & (data.date == match_date)
        ].reset_index()
        if player_data.shape[0] != 0 and not np.isnan(player_data.label[0]):
            scores.append((player_data.label[0], player_data.player_id[0]))
    scores = sorted(scores, reverse=True)
    # print(scores)
    return ([x[0] for x in scores[:11]], [x[1] for x in scores[:11]])


def get_predicted_score(predicted_players, match_date, match_type):
    scores = []
    data = pd.read_csv(f"../../../data/processed/formatwise/{match_type}.csv")
    for player_id in predicted_players:
        player_data = data[
            (data.player_id == player_id) & (data.date == match_date)
        ].reset_index()
        if player_data.shape[0] and not np.isnan(player_data.label[0]):
            scores.append(player_data.label[0])
    return sorted(scores, reverse=True)


def predict(
    team1: str,
    team2: str,
    team_players1: List[Dict[str, Any]],
    team_players2: List[Dict[str, Any]],
    match_date: str,
    match_type: str,
) -> List[str]:
    """
    Highly optimized prediction function using parallel processing and caching.
    """
    all_players = team_players1 + team_players2
    player_args = [(p["id"], match_date, match_type) for p in all_players]

    actual_scores, actual_players = get_actual_scores(
        team_players1, team_players2, match_date, match_type
    )

    # Process players in parallel
    with ThreadPoolExecutor(max_workers=min(32, len(all_players))) as executor:
        future_to_player = {
            executor.submit(calculate_features_batch, args): args
            for args in player_args
        }

        # Collect results as they complete
        player_scores = {}
        for future in as_completed(future_to_player):

            player_id, features = future.result()
            player_scores[player_id] = features["score"][0]

    # Sort and return top players efficiently
    sorted_players = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
    # print([score for player_id, score in sorted_players[:11]])
    overall_mae = -1
    num_common = 0

    predicted_players = [player_id for player_id, _ in sorted_players[:11]]

    output_list = [(p["name"], p["id"], player_scores[p["id"]], match_date, match_type) for p in all_players if p["id"] in predicted_players]
    
    output_players = pd.DataFrame(
        output_list,
        columns=[
            "Player_Name",
            "Player_ID",
            "Predicted_Score",
            "Match_Date",
            "Match_Type",
        ],
    )
    output_players.to_csv(f"{match_type}_{match_date}.csv", index=False)
    if len(actual_scores) > 0:
        predicted_scores = get_predicted_score(
            predicted_players, match_date, match_type
        )
        predicted_sum = np.sum(predicted_scores)
        actual_sum = np.sum(actual_scores)

        union_players = set(predicted_players + actual_players)
        num_common = len(predicted_players) + len(actual_players) - len(union_players)

        print(actual_scores)
        print(predicted_scores)
        print("Number of Common Players:", num_common)

        overall_mae = np.abs(predicted_sum - actual_sum)

    return (predicted_players, overall_mae, num_common)


def model_inference(last_row, match_type):

    # Convert last_row (Series) to DataFrame
    features_df = last_row.to_frame().T  # Transpose to get a DataFrame

    # Define the feature columns
    cat_1_columns = [
        "boundaries",
        "sixes",
        "fifties",
        "hundreds",
        "ducks",
        "thirty_run_innings",
        "caught",
        "run out",
        "direct",
        "stumped",
        "3+catches",
        "wickets_taken",
        "3wickets_haul",
        "5wickets_haul",
        "maiden_overs",
        "wickets_lbw_bowled",
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

    numerical_features = []
    # Construct feature names as per training
    window = 30
    for col in cat_1_columns:
        col_name = f"{col}_hcma_w{window}"
        numerical_features.append(col_name)

    window = 5
    alpha = 0.7
    for col in cat_2_columns:
        col_name = f"{col}_ewma_w{window}_alpha{alpha}"
        numerical_features.append(col_name)

    # Ensure that all necessary features are present
    missing_cols = set(numerical_features) - set(features_df.columns)
    if missing_cols:
        print(f"Warning: Missing columns in input data: {missing_cols}")
        # Handle missing columns (e.g., fill with zeros)
        for col in missing_cols:
            features_df[col] = 0  # Adjust as necessary for your use case

    scaler_fit = joblib.load(f"../../../data/interim/scaler/scaler_{match_type}.save")

    features_df[numerical_features] = scaler_fit.transform(
        features_df[numerical_features]
    )
    # Select and prepare the feature columns
    X = features_df[numerical_features].copy()
    X = X.astype(float)

    # Convert to torch tensor
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    # Load the ANN model
    # Placeholder for model loading
    # Adjust input_size and other parameters according to your model
    input_size = X_tensor.shape[1]
    model = LargeNet(input_size=input_size)
    checkpoint = torch.load(get_model_path(match_type))
    model.load_state_dict(checkpoint)  # Load model weights
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(
            X_tensor
        )  # Get raw model outputs, which are the predicted scores
    predicted_scores = outputs  # These are your predicted regression scores
    # Return the prediction
    return predicted_scores


# Optional: Function to clear caches if memory becomes a concern
def clear_caches():
    FEATURE_CACHE.clear()
    PLAYER_DATA_CACHE.clear()
