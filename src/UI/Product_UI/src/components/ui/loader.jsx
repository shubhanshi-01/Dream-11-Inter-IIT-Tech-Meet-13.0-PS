import React from "react";
import Spinner from "./spinner";

const Loader = ({ isVisible, children }) => {
  return (
    <div
      className="flex flex-col items-center justify-center h-screen w-screen fixed z-50 bg-[#000000b9] left-0 top-0 "
      style={{
        display: isVisible ? "flex" : "none",
      }}
    >
      <Spinner className="w-12 h-12 text-gray-200 dark:text-gray-600 fill-blue-600" />
      <div
        className="text-white text-lg"
        style={{
          fontFamily: "Poppins",
        }}
      >
        {children}
      </div>
    </div>
  );
};

export default Loader;
