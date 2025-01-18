import React, { useState, useEffect } from "react";
import axios from "axios";

const TeamModal = ({
  isVisible,
  setIsVisible,
  selectedTeam,
  otherTeam,
  setSelectedTeam,
  setPlayersModalVisible,
  setCurrentTeam,
}) => {
  const [searchQuery, setSearchQuery] = useState(selectedTeam);
  const [recommendations, setRecommendations] = useState([]);
  const [teams, setTeams] = useState([]);
  useEffect(() => {
    axios
      .get("/api/get_teams")
      .then((res) => {
        console.log(res.data);
        setTeams(res.data.teams);
      })
      .catch((err) => {
        console.log(err);
        alert("Error fetching teams.");
      });
    setSearchQuery(selectedTeam);
  }, [selectedTeam]);

  useEffect(() => {
    if (isVisible && selectedTeam) {
      SelectPlayers();
    }
  }, [isVisible, selectedTeam]);

  const handleBackgroundClick = (e) => {
    // Prevent event from propagating to child elements
    if (e.target === e.currentTarget) {
      setIsVisible(false);
    }
  };

  const handleSearch = () => {
    // Only show recommendations if search term is 3 or more characters
    const value = searchQuery;
    if (value.length >= 1) {
      const filteredTeam = teams.filter(
        (team) =>
          team?.toLowerCase().includes(value.toLowerCase()) &&
          team !== otherTeam
      );
      setRecommendations(filteredTeam);
    } else {
      setRecommendations([]);
    }
  };

  useEffect(() => {
    handleSearch();
  }, [searchQuery]);

  // const filteredTeams = teams.filter((team) => {
  //   team?.toLowerCase().includes(searchQuery.toLowerCase());
  // });

  const handleSelectTeam = (team) => {
    setSelectedTeam(team);
    setSearchQuery("");
    setRecommendations([team]);
  };

  const SelectPlayers = () => {
    if (selectedTeam === "") {
      alert("Please select a team");
      return;
    }
    setPlayersModalVisible(true);
    setCurrentTeam(selectedTeam);
    setIsVisible(false);
  };

  if (!isVisible) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
      onClick={handleBackgroundClick}
    >
      <div className="bg-white w-96 p-6 rounded-lg shadow-lg">
        <h2 className="text-xl font-light mb-4">Select Team</h2>
        <input
          type="text"
          className="w-full p-2 mb-4 border border-gray-300 rounded-lg"
          placeholder="Search teams..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value);
          }}
        />
        <ul className="space-y-2 overflow-auto max-h-[40vh]">
          {recommendations.length > 0 ? (
            recommendations?.map((team) => (
              <li
                key={team}
                className={`cursor-pointer p-2 bg-gray-100 rounded-lg hover:bg-gray-200 ${
                  team == selectedTeam
                    ? "bg-red-200 text-black hover:bg-red-100"
                    : ""
                }`}
                onClick={() => handleSelectTeam(team)}
              >
                {team}
              </li>
            ))
          ) : (
            <li className="text-gray-500">Search For a Team to Select</li>
          )}
        </ul>

        <button
          className={`mt-4 w-full  p-2 rounded-lg ${
            selectedTeam === ""
              ? "bg-gray-300 text-gray-400"
              : "bg-red-500 text-white"
          }
            `}
          onClick={SelectPlayers}
        >
          Select Players
        </button>
      </div>
    </div>
  );
};

export default TeamModal;
