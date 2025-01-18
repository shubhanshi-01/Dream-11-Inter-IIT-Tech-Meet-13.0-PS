import React, { useState, useEffect } from "react";
import axios from "axios";
import { Button } from "./button";
import { X } from "lucide-react";
import { PlayerImage } from "./PlayerImage";
import Spinner from "./spinner";

const PlayersModal = ({
  isVisible,
  setIsVisible,
  selectedPlayers,
  setSelectedPlayers,
  currentTeam,
  autofilledPlayers,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!currentTeam) return;

    console.log(recommendations);
    console.log("Fetching players for team", currentTeam);
    if (selectedPlayers.length < 11) {
      setLoading(true);
    }
    axios
      .post("/api/get_players", {
        team_name: currentTeam,
      })
      .then((res) => {
        setLoading(false);

        const data = res.data;
        console.log(data);
        setPlayers(data.players);
        setRecommendations(data.players);
      });
  }, [currentTeam, autofilledPlayers]);

  const handleBackgroundClick = (e) => {
    // Prevent event from propagating to child elements
    if (e.target === e.currentTarget) {
      setRecommendations([]);
      setIsVisible(false);
    }
  };

  const handlePlayerSelect = (player) => {
    setSearchTerm("");

    setRecommendations(
      players.filter((p) => !selectedPlayers.includes(p) && player != p)
    );

    setSelectedPlayers((currentPlayers) => [...currentPlayers, player]);
  };
  const handleSearch = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    // Only show recommendations if search term is 3 or more characters

    const filteredPlayers = players.filter(
      (player) =>
        (player.name.toLowerCase().includes(value.toLowerCase()) ||
          player.alt_name.toLowerCase().includes(value.toLowerCase())) &&
        !selectedPlayers.includes(player)
    );
    setRecommendations(filteredPlayers);
  };
  const handleContentClick = (e) => {
    // Prevent closing when clicking inside the modal
    e.stopPropagation();
  };

  const handleDone = () => {
    setRecommendations([]);
    console.log("resetting Recommendations");
    console.log(recommendations);
    setIsVisible(false);
    setSearchTerm("");
  };

  if (!isVisible) return null;
  return (
    <div
      className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-10"
      onClick={handleBackgroundClick}
    >
      <div
        className="bg-white rounded-lg shadow-xl w-[min(90%,700px)] p-6"
        onClick={handleContentClick}
      >
        <div className="flex items-center mb-4">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="mr-3 text-gray-500"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <h2 className="text-xl font-semibold text-gray-800">Search</h2>
        </div>
        <input
          type="text"
          placeholder={
            selectedPlayers.length >= 11
              ? "11 Players already selected"
              : "Search players to add..."
          }
          value={searchTerm}
          onChange={handleSearch}
          className={`w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            selectedPlayers.length >= 11 ? "bg-gray-300" : "bg-white"
          }`}
          disabled={selectedPlayers.length >= 11}
        />
        {loading && <Spinner className="m-5"></Spinner>}
        {recommendations.length > 0 && selectedPlayers.length < 11 && (
          <ul
            className="z-10 w-full bg-white border border-gray-300 rounded-md mt-1 shadow-lg max-h-60 overflow-y-auto"
            style={{
              display: loading ? "none" : "block",
            }}
          >
            {recommendations.map((player) => (
              <li
                key={player.id}
                onClick={() => handlePlayerSelect(player)}
                className="px-3 py-2 hover:bg-gray-100 cursor-pointer flex items-center select-none"
              >
                <PlayerImage
                  src={player.image}
                  alt={player.name}
                  className="w-10 h-10 rounded-full mr-3 object-contain"
                  fallbackPadding="2px"
                />
                {player.name}
              </li>
            ))}
          </ul>
        )}
        <div className="mt-4">
          <div style={{ flexDirection: "row", display: "flex" }}>
            <h3 className="text-lg font-semibold text-gray-800 flex-1">
              Selected Players
            </h3>
            <h3 className="text-lg font-semibold text-gray-800 ">
              {selectedPlayers.length}/11
            </h3>
          </div>
          <ul
            className={`mt-2 flex flex-wrap justify-center overflow-auto ${
              selectedPlayers.length >= 11 ? "max-h-[50vh]" : "max-h-[25vh]"
            }`}
          >
            {[]
              .concat(selectedPlayers)
              .reverse()
              .map((player, index) => (
                <div
                  style={{ position: "relative", margin: 5 }}
                  className="items-center select-none"
                  key={index}
                >
                  <li
                    key={player.name}
                    style={{
                      width: 120,
                      height: 160,
                      borderRadius: 8,
                      padding: "10px",
                      boxShadow: "0 0 10px #00000010",
                    }}
                  >
                    <PlayerImage
                      src={player.image}
                      alt={player.name}
                      className="w-full h-70% pt-5 rounded-full mr-3 object-contain"
                    />
                    <div
                      style={{
                        alignContent: "center",
                        justifyContent: "center",
                        textAlign: "center",
                      }}
                    >
                      {player.name}
                    </div>
                  </li>

                  <X
                    style={{
                      position: "absolute",
                      top: 0,
                      right: 0,
                      marginTop: 5,
                      marginRight: 5,
                      opacity: 0.3,
                    }}
                    className="cursor-pointer align-middle justify-center"
                    onClick={() =>
                      setSelectedPlayers(
                        selectedPlayers.filter((p) => p !== player)
                      )
                    }
                  />
                </div>
              ))}
          </ul>

          <Button
            className="mt-4 w-full bg-red-500 text-white p-2 rounded-lg"
            onClick={() => handleDone()}
          >
            Done
          </Button>
        </div>
      </div>
    </div>
  );
};
export default PlayersModal;
