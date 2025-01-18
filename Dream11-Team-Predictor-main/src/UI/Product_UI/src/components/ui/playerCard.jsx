import { Award, Crown } from "lucide-react";
import Image from "next/image";
import React, { useEffect, useState } from "react";
import { PlayerImage } from "./PlayerImage";

const PlayerCard = ({ name, teamName, imageUrl, playerType, playerRole }) => {
  return (
    <>
      <div className="group relative w-[150px] h-[150px] rounded-xl overflow-hidden transition-all duration-300 hover:scale-105 cursor-pointer ">
        {/* Glassmorphism background */}
        <div className="absolute inset-0 bg-white/30 backdrop-blur-lg border border-white/20 shadow-xl transition-all duration-300 group-hover:bg-white/20" />
        <span
          className="absolute right-2 top-2 px-3 py-1 rounded-full bg-[#ffc800] text-[12px] font-bold text-[#554713]"
          style={{
            fontFamily: "Poppins",
            display: playerType.toLowerCase() == "player" ? "none" : "block",
          }}
        >
          {playerType.toLowerCase() == "captain"
            ? "C"
            : playerType.toLowerCase() == "vice captain"
            ? "VC"
            : ""}
        </span>
        <span
          className="absolute right-0 top-[0px] z-50 p-2 rounded-bl-lg opacity-70 bg-[#fcfcfc44] text-[12px] font-bold text-[#554713] flex flex-row gap-1"
          style={{
            fontFamily: "Poppins",
            display: playerType.toLowerCase() != "player" ? "none" : "flex",
          }}
        >
          {(playerRole == "all-rounder" || playerRole == "batsman") && (
            <Image
              src="/cricket-bat.png"
              width="100"
              height="100"
              className="w-4 h-4 opacity-[60%]"
              alt="cricket bat"
            />
          )}
          {(playerRole == "all-rounder" || playerRole == "bowler") && (
            <Image
              src="/cricket-ball.png"
              width="100"
              height="100"
              className="w-4 h-4  opacity-[60%]"
              alt="cricket ball"
            />
          )}
        </span>
        {/* Content container */}
        <div className="relative h-full flex flex-col items-center p-4 px-0 pb-0">
          {/* Image container */}
          <div className="w-full h-full pb-[1.8em] overflow-hidden transition-transform duration-300 group-hover:scale-105">
            <PlayerImage
              src={imageUrl}
              alt="Player"
              width="500"
              height="500"
              className="w-full h-full object-contain"
            />
          </div>

          {/* Player info */}
          <div
            className="absolute bottom-[-1.8em] text-center  bg-[#1c465f] w-full mb-0 mt-0 z-30 p-2  group-hover:bottom-0 "
            style={{
              transition: "0.5s",
            }}
          >
            <h2 className="text-[18px] font-light text-white transition-all duration-300 ">
              {name}
            </h2>

            <div className="text-white/70">
              <p className="font-medium text-[14px]">{teamName}</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default PlayerCard;
