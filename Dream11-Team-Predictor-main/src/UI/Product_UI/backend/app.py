from flask import Flask, jsonify, request
import json
import pandas as pd
import time
import random
from flask_cors import CORS, cross_origin
from transform_data import transform_match_data
from LLM import get_reasons
import sys
import os

sys.path.append(os.path.abspath("../../../model/"))
from predict_model import predict

sys.path.append(os.path.abspath("../UI/Product_UI/backend/"))


import time

app = Flask(__name__)
cors = CORS(app)  # allow CORS for all domains on all routes.
app.config["CORS_HEADERS"] = "Content-Type"


# Path to the files
TEAMS_FILE = "data/teams.json"
PLAYERS_FILE = "data/team_players.json"


@app.route("/get_teams/", methods=["GET"])
@cross_origin()
def get_team():
    """Returns a list of team names."""
    try:
        with open(TEAMS_FILE, "r") as file:
            data = json.load(file)
        return jsonify({"teams": data.get("teams", [])}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_players/", methods=["POST"])
@cross_origin()
def get_players():
    """Returns a list of players for a given team."""
    try:
        # Get the team name from the request
        data = request.get_json()
        team_name = data.get("team_name", "").strip()
        print("Getting players for:", team_name)

        final_people = pd.read_csv("data/final_people.csv")

        if not team_name:
            return jsonify({"error": "Team name is required"}), 400

        player_ids = []
        with open(PLAYERS_FILE, "r") as file:
            file_data = json.load(file)
            player_ids = file_data.get(team_name, [])
        if not player_ids:
            return jsonify({"error": f"No players found for team '{team_name}'"}), 404
        players = []
        for player_id in player_ids:
            player = final_people[final_people["identifier"] == player_id].fillna("")
            if player.empty:
                continue
            players.append(
                {
                    "id": player_id,
                    "name": player["name"].values[0],
                    "alt_name": player["alt_name"].values[0],
                    "image": "/player_images/" + player_id + ".jpg",
                }
            )
        return jsonify({"players": players}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_matches", methods=["POST"])
def get_matches():
    request_params = request.get_json()
    # Get query parameters
    match_date = request_params.get("matchDate")
    match_type = request_params.get("matchType")
    print(match_date, match_type)
    print("Finding matches for", match_type, match_date)
    # Load matches data
    with open("data/matches_data.json", "r") as f:
        data = json.load(f)

    # Return empty list if date not found
    if match_date not in data["matches"]:
        # match_date = "2024-11-22"
        # match_type = "Test"
        return (
            jsonify({"matches": [], "matchDate": match_date, "matchType": match_type}),
            200,
        )

    # Return matches for given date and type
    matches = data["matches"][match_date].get(match_type, [])

    # Convert matches to required format
    formatted_matches = [match for match in matches]

    return (
        jsonify(
            {
                "matches": formatted_matches,
                "matchDate": match_date,
                "matchType": match_type,
            }
        ),
        200,
    )


@app.route("/run_inference/", methods=["POST"])
@cross_origin()
def run_inference():
    """
    Runs an ML inference for a match.
    Expected input:
    - team_1: Name of the first team
    - team_2: Name of the second team
    - team_1_players: List of 11 players from the first team
    - team_2_players: List of 11 players from the second team
    - match_date: Date of the match
    - match_type: Type of the match (e.g., Test, ODI, T20)
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = [
            "team_1",
            "team_2",
            "team_1_players",
            "team_2_players",
            "match_date",
            "match_type",
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"'{field}' is required"}), 400

        if len(data["team_1_players"]) != 11 or len(data["team_2_players"]) != 11:
            return jsonify({"error": "Each team must have exactly 11 players"}), 400

        print("Starting prediction...")
        time_start = time.time()
        selected_players, overall_mae, num_common = predict(
            data["team_1"],
            data["team_2"],
            data["team_1_players"],
            data["team_2_players"],
            data["match_date"],
            data["match_type"],
        )
        print("Overall Team MAE Diff:", overall_mae)
        time_predicted = time.time()
        print("Prediction completed in", time_predicted - time_start, "seconds")

        formatted_data = transform_match_data(
            data["team_1"],
            data["team_2"],
            data["team_1_players"],
            data["team_2_players"],
            data["match_date"],
            data["match_type"],
            selected_players,
        )

        time_formatted = time.time()
        # reasons = get_reasons(formatted_data)
        # time_LLM = time.time()
        # print("Reasons generated in", time_LLM - time_formatted, "seconds")

        player_response = []
        combined_players = data["team_1_players"] + data["team_2_players"]
        for i in range(len(formatted_data["players"])):
            if formatted_data["players"][i]["selected"]:
                player_data = {}
                player_data["id"] = formatted_data["players"][i]["player_id"]
                if player_data["id"] == selected_players[0]:
                    player_data["player_type"] = "Captain"
                elif player_data["id"] == selected_players[1]:
                    player_data["player_type"] = "Vice Captain"
                else:
                    player_data["player_type"] = "Player"
                player_data["name"] = formatted_data["players"][i]["player_name"]
                player_data["team"] = formatted_data["players"][i]["team"]
                player_data["stats"] = formatted_data["players"][i]["player_stats"]
                player_data["reason"] = "Loading..."
                for j in range(len(combined_players)):
                    if (
                        combined_players[j]["id"]
                        == formatted_data["players"][i]["player_id"]
                    ):
                        player_data["image"] = combined_players[j]["image"]
                        break
                # for j in range(len(reasons["reasons"])):
                #     if (
                #         reasons["reasons"][j]["player_id"]
                #         == formatted_data["players"][i]["player_id"]
                #     ):
                #         player_data["reason"] = reasons["reasons"][j]["reason"]
                #         break
                player_response.append(player_data)

        final_response = {
            "players": player_response,
            "match_date": data["match_date"],
            "match_type": data["match_type"],
            "team_1": data["team_1"],
            "team_2": data["team_2"],
            "raw_formatted_data": formatted_data,
            "overall_mae": overall_mae,
            "num_common": num_common,
        }

        time_end = time.time()
        print("Total Time taken for player Response", time_end - time_start, "seconds")
        # Return simulated response
        return jsonify(final_response), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route("/formulate_reasons/", methods=["POST"])
@cross_origin()
def formulate_reasons():
    """
    Based on player stats, generates reasons for player selection in natural language.
    Expected input:
    - data: the returned data output from run_inference/
    """
    try:
        time_start = time.time()
        data = request.get_json()
        formatted_data = data.get("raw_formatted_data")
        reasons = get_reasons(formatted_data)
        print(reasons)
        for i in range(len(data["players"])):
            for j in range(len(reasons["reasons"])):
                if data["players"][i]["id"] == reasons["reasons"][j]["player_id"]:
                    data["players"][i]["reason"] = reasons["reasons"][j]["reason"]

        return jsonify(data), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
