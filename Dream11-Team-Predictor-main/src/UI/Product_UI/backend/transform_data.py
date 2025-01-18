import random
from get_stats import get_stats


def transform_match_data(
    team_1: str,
    team_2: str,
    team_1_players,
    team_2_players,
    match_date: str,
    match_type: str,
    selected_players,  # List of selected player ids
):
    """
    Transform match data into the required format with player statistics

    Args:
        team_1: Name of first team
        team_2: Name of second team
        team_1_players: List of players from team 1
        team_2_players: List of players from team 2
        match_type: Type of match (Test/ODI/T20/etc)
        selected_players: List of selected player IDs

    Returns:
        Dictionary containing formatted match data with player statistics
    """

    all_players = []
    # Process team 1 players
    for player in team_1_players:
        player_data = {
            "player_id": player["id"],  # Generate unique ID
            "player_name": player["name"],
            "team": team_1,
            "selected": player["id"] in selected_players,
            "player_stats": get_stats(player["id"], match_date, match_type),
        }
        all_players.append(player_data)
    # Process team 2 players
    for player in team_2_players:
        player_data = {
            "player_id": player["id"],  # Generate unique ID
            "player_name": player["name"],
            "team": team_2,
            "selected": player["id"] in selected_players,
            "player_stats": get_stats(player["id"], match_date, match_type),
        }
        all_players.append(player_data)
    # Create final formatted data
    formatted_data = {"match_type": match_type, "players": all_players}
    return formatted_data
