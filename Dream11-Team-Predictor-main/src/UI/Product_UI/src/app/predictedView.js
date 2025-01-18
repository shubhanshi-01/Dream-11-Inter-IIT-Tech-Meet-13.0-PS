import React, { useState, useEffect } from "react";
import PlayerCard from "@/components/ui/playerCard";
import LeftPanel from "@/components/ui/LeftPanel";
import PlayerSheet from "@/components/ui/playerSheet";

export const PredictedView = ({
  players = [],
  matchDate,
  matchType,
  teamA,
  teamB,
  overall_mae,
  num_common,
}) => {
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const priorities = { "all-rounder": 1, batsman: 2, bowler: 3, new: 4 };
  players.sort((a, b) => {
    if (a.player_type == "Captain") {
      return -1;
    } else if (b.player_type == "Captain") {
      return 1;
    } else {
      if (a.player_type == "Vice Captain") {
        return -1;
      } else if (b.player_type == "Vice Captain") {
        return 1;
      } else {
        return (
          priorities[a.stats.player_role] - priorities[b.stats.player_role]
        );
      }
    }
  });
  console.log(players);
  const handlePlayerClick = (player) => {
    setSelectedPlayer(player);
  };
  useEffect(() => {
    console.log("overall MAE", overall_mae);
    console.log("num common", num_common);

    for (let player of players) {
      console.log(player);
      if (player.id == selectedPlayer?.id) {
        setSelectedPlayer(player);
      }
    }
  }, [players]);
  return (
    <div className="relative flex w-full min-h-screen overflow-auto">
      {/* <LeftPanel teams={teams} className="z-10" /> */}

      <div className="relative w-full h-full flex flex-wrap justify-center items-center content-center pt-10">
        <div className="absolute z-10 left-0 top-0 w-full flex justify-between items-center px-10 py-6 bg-[#1c465f]">
          <div
            className="hidden md:block text-md  absolute top-2 right-[calc(50%+200px)] text-[#afe2ff]"
            style={{
              fontFamily: "Poppins",
            }}
          >
            {matchType} &#8226; {matchDate}
          </div>

          <div className="text-xl md:text-2xl font-thin absolute z-10 left-1/2 top-0 -translate-x-1/2 p-5 px-10 bg-[#c14343] text-[#ffffff] rounded-b-lg text-center">
            Your Dream Team
          </div>

          <div
            className="hidden md:block text-md absolute top-2 left-[calc(50%+200px)] text-[#afe2ff]"
            style={{
              fontFamily: "Poppins",
            }}
          >
            {teamA} vs {teamB}
          </div>
        </div>

        <div className="flex flex-wrap gap-10 w-full min-h-full justify-center items-center align-middle content-center max-w-[1024px] mt-[80px]">
          {players.map((player, index) => (
            <div key={index} onClick={() => handlePlayerClick(player)}>
              <PlayerCard
                name={player.name}
                teamName={player.team}
                imageUrl={player.image}
                playerType={player.player_type}
                playerRole={player.stats.player_role}
              />
            </div>
          ))}
        </div>
      </div>

      {overall_mae != -1 && (
        <div
          className="absolute z-10 bottom-0 left-1/2 -translate-x-1/2 p-3 hidden sm:flex justify-center items-center gap-10 flex-row w-full bg-[#1c465f] font-medium text-[#63c6ff]"
          style={{ fontFamily: "Poppins" }}
        >
          <div className="text-lg  font-medium">
            Overall Team MAE:
            <span className="font-bold text-white"> {overall_mae}</span>
          </div>
          <div className="text-lg font-medium">
            Common Players:
            <span className="font-bold text-white "> {num_common}</span>
          </div>
        </div>
      )}

      <PlayerSheet
        player={selectedPlayer}
        open={!!selectedPlayer}
        onOpenChange={(open) => !open && setSelectedPlayer(null)}
      />
    </div>
  );
};

export default PredictedView;
