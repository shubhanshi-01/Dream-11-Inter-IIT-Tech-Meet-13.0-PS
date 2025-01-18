"use client";
import React, { useEffect, useState } from "react";
import Image from "next/image";

const LeftPanel = ({ teams = ["Team A", "Team B"] }) => {
  return (
    <div
      className="cardContainer w-[350px] h-full rounded-r-xl py-[150px] px-5"
      style={{
        background: "rgba( 255, 255, 255, 0.2 )",
        boxShadow: "0 8px 32px 0 rgba( 31, 38, 135, 0.37 )",
        backdropFilter: "blur( 20.5px )",
        WebkitBackdropFilter: "blur( 20.5px )",
        border: "1px solid rgba( 255, 255, 255, 0.38 )",
        fontFamily: "Poppins",
      }}
    >
      <div className="logo absolute left-2 top-2">
        <Image src="/logo.png" width="180" height="180" alt=" "></Image>
      </div>
      <div className="flex justify-evenly items-center font-medium text-xl text-center">
        <div className="text-center flex flex-wrap flex-col justify-center items-center">
          <div className="w-[50px] h-[50px] rounded-full bg-[#0000002f]"></div>

          <span>{teams[0]}</span>
        </div>
        <span className="w-[100px] h-[100px] text-md text-zinc-500 italic text-center">
          vs
        </span>
        <div className="text-center flex flex-wrap flex-col justify-center items-center">
          <div className="w-[50px] h-[50px] rounded-full bg-[#0000002f]"></div>
          <span>{teams[1]}</span>
        </div>
      </div>
    </div>
  );
};
export default LeftPanel;
