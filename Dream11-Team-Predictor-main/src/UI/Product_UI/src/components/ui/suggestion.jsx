import Image from "next/image";
import React from "react";
import Link from "next/link";
import { ArrowRight, Calendar } from "lucide-react";
export const SuggestionCard = ({
  teamA,
  teamB,
  matchType,
  matchDate,
  onClick,
}) => {
  return (
    <div
      className="relative w-full md:w-auto min-w-[300px] min-h-[200px] p-10 rounded-lg cursor-pointer text-left text-black bg-[#ffffffa8] overflow-hidden group transition-transform hover:translate-y-[-10px]"
      style={{
        fontFamily: "Poppins",
        backdropFilter: "blur(30px)",
        boxShadow: "0 10px 10px #00000042",
      }}
      onClick={onClick}
    >
      <div className="absolute right-1 top-1 -rotate-45 transition-transform group-hover:rotate-0">
        <ArrowRight size={30} className="text-[#00000066]"></ArrowRight>
      </div>
      <div
        className="absolute -top-[0%] -right-[50%] w-full h-full bg-[#ffffff3a]"
        style={{ zIndex: "-1" }}
      ></div>
      <div className="main text-lg w-full">
        <div className="w-full ">{teamA} </div>
        <div className="font-bold">vs</div>{" "}
        <div className="w-full ">{teamB}</div>
      </div>
      <div className="text-sm flex justify-between mt-5 text-[#000000] font-bold opacity-70 flex-wrap gap-2 absolute bottom-5 w-full left-0 px-4">
        <div className="px-3 bg-[#db4b4bb5] text-[white] font-light rounded-full">
          {matchType}
        </div>
        <div className="text-sm font-light">
          <Calendar
            size={18}
            className="text-sm inline-block ml-1 -mt-1"
          ></Calendar>{" "}
          {matchDate}
        </div>
      </div>
    </div>
  );
};
