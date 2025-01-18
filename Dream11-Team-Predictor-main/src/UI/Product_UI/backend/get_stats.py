import pandas as pd
import numpy as np
import os


def convert_to_native_types(value):
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32)):
        return float(value)
    elif isinstance(value, pd.Timestamp):
        return value.strftime("%d-%m-%Y")
    elif isinstance(value, dict):
        return {k: convert_to_native_types(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_native_types(item) for item in value]
    return value


def calculate_player_statistics(group, player_role):
    # Aggregate statistics for the player
    matches = group["match_id"].nunique()
    total_runs = group["total_runs"].sum()
    dismissals = group["is_out"].sum()
    batting_avg = total_runs / dismissals if dismissals > 0 else -1
    balls_faced = group["balls_faced"].sum()
    strike_rate = (total_runs / balls_faced * 100) if balls_faced > 0 else -1
    centuries = group["hundreds"].sum()
    half_centuries = group["fifties"].sum()

    # Bowling stats
    wickets = group["wickets_taken"].sum()
    runs_conceded = group["runs_conceded"].sum()
    balls_bowled = group["balls_bowled"].sum()
    bowling_avg = runs_conceded / wickets if wickets > 0 else -1
    economy_rate = runs_conceded / (balls_bowled / 6) if balls_bowled > 0 else -1

    # Recent forms (last 5 matches or fewer)
    recent_form = (
        group.tail(5)
        .apply(
            lambda row: {
                "match": {"opponent": row["opponent"], "date": row["date"]},
                "runs": f"{row['total_runs']} ({row['balls_faced']})",
                "wickets": (
                    f"{row['runs_conceded']}/{row['wickets_taken']}"
                    if row["wickets_taken"] > 0
                    else f"{row['runs_conceded']}/-"
                ),
            },
            axis=1,
        )
        .tolist()
    )

    # Construct the output dictionary
    player_stats = {
        "matches": matches,
        "batting_average": batting_avg,
        "runs": total_runs,
        "batting_avg": batting_avg,
        "strike_rate": strike_rate,
        "centuries": centuries,
        "half_centuries": half_centuries,
        "bowling_avg": bowling_avg,
        "economy_rate": economy_rate,
        "recent_form": recent_form,
        "player_role": player_role,
    }

    return convert_to_native_types(player_stats)


def get_stats(player_id, date, match_type):
    format_mapping = {"it20": "t20", "mdm": "test", "odm": "odi"}
    input_folder = "../../../data/processed/playerwise/"
    file_path = os.path.join(input_folder, f"{player_id}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Player data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Normalize 'match_type' to lowercase and map formats
    if "match_type" in df.columns:
        df["match_type"] = df["match_type"].str.lower()
        df["revised_format"] = (
            df["match_type"].map(format_mapping).fillna(df["match_type"])
        )
    else:
        raise ValueError("match_type column not found in the data.")

    # Convert 'date' column to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        date = pd.to_datetime(date)

        # Filter by date - only data before the given date
        df = df[df["date"] < date]
    else:
        raise ValueError("Date column not found in the data; cannot filter by date.")

    # Now filter by 'revised_format' matching the given 'match_type' (after mapping)
    match_type_lower = match_type.lower()
    match_type_mapped = format_mapping.get(match_type_lower, match_type_lower)
    df = df[df["revised_format"] == match_type_mapped]

    # Sort by date
    df.sort_values("date", inplace=True)
    total_matches = len(df)
    bowled_matches = df[df["balls_bowled"] > 0].shape[0]
    batted_matches = df[df["balls_faced"] > 0].shape[0]

    if total_matches == 0:
        player_role = "new"
    else:
        # Performance metrics for bowling
        wickets_per_match = df["wickets_taken"].sum() / total_matches
        bowling_avg = (
            df["runs_conceded"].sum() / df["wickets_taken"].sum()
            if df["wickets_taken"].sum() > 0
            else float("inf")
        )
        economy_rate = (
            df["runs_conceded"].sum() / (df["balls_bowled"].sum() / 6)
            if df["balls_bowled"].sum() > 0
            else float("inf")
        )

        # Performance metrics for batting
        batting_avg = (
            df["total_runs"].sum() / df["is_out"].sum()
            if df["is_out"].sum() > 0
            else -1
        )
        strike_rate = (
            (df["total_runs"].sum() / df["balls_faced"].sum() * 100)
            if df["balls_faced"].sum() > 0
            else -1
        )
        half_centuries = df["fifties"].sum()

        # Role determination based on enhanced criteria
        is_bowler = bowled_matches / total_matches >= 0.3 and (
            wickets_per_match >= 1.0 or bowling_avg <= 35.0 or economy_rate <= 7.0
        )
        is_batsman = batted_matches / total_matches >= 0.3 and (
            batting_avg >= 25.0 or strike_rate >= 120.0 or half_centuries >= 2
        )

        if is_bowler and is_batsman:
            player_role = "all-rounder"
        elif is_bowler:
            player_role = "bowler"
        elif is_batsman:
            player_role = "batsman"
        else:
            player_role = "new"

        # print(player_role)

    # Calculate and return statistics
    final = calculate_player_statistics(df, player_role)
    return final
