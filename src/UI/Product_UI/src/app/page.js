"use client";
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import gsap from "gsap";
import { Plus, PlusCircle, Cross } from "lucide-react";
import { PredictedView } from "./predictedView";
import TeamModal from "@/components/ui/TeamModal";
import PlayersModal from "@/components/ui/PlayersModal";
import { options } from "@/data/matchTypes.json";
import axios from "axios";
import { SuggestionCard } from "@/components/ui/suggestion";
import Loader from "@/components/ui/loader";
import { translateText } from "@/lib/translate.js";

const StadiumViewer = () => {
  const initialScale = 0.2;
  const [rotationSpeed, setRotationSpeed] = useState(2);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [step, setStep] = useState(0);
  const [scale, setScale] = useState(initialScale);
  const [position, setPosition] = useState({ x: 0, y: -5, z: 0 });
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
  const [backgroundColor, setBackgroundColor] = useState("#75bca1");
  const [teamModalVisibleA, setTeamsModalVisibleA] = useState(false);
  const [teamModalVisibleB, setTeamsModalVisibleB] = useState(false);
  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");
  const [currentTeam, setCurrentTeam] = useState("");
  const [playersModalVisibleA, setPlayersModalVisibleA] = useState(false);
  const [playersModalVisibleB, setPlayersModalVisibleB] = useState(false);
  const [selectedA, setSelectedA] = useState(new Set([]));
  const [selectedB, setSelectedB] = useState(new Set([]));
  const [matchType, setMatchType] = useState("");
  const [matchDate, setMatchDate] = useState("");
  const [prediction, setPrediction] = useState({});
  const [suggestedMatches, setSuggestedMatches] = useState({});
  const [loading, setLoading] = useState(false);
  const [teamPredicted, setTeamPredicted] = useState(false);

  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const modelRef = useRef(null);
  const controlsRef = useRef(null);
  const animationRef = useRef(null);
  const colorRef = useRef(new THREE.Color(backgroundColor));

  //emptying players if team changed
  useEffect(() => {
    setSelectedA([]);
  }, [teamA]);
  useEffect(() => {
    setSelectedB([]);
  }, [teamB]);

  // getting suggestions for matches
  useEffect(() => {
    if (!matchDate || !matchType) return;
    console.log(new Date(matchDate).toISOString().slice(0, 10), matchType);
    axios
      .post("/api/get_matches", {
        matchDate: new Date(matchDate).toISOString().slice(0, 10),
        matchType,
      })
      .then((res) => {
        console.log(res.data);
        setSuggestedMatches(res.data);
      })
      .catch((err) => {
        console.log(err);
        alert("Error fetching suggested matches.");
      });
  }, [matchType, matchDate]);

  // 3d Model setup
  useEffect(() => {
    // Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    scene.background = new THREE.Color(backgroundColor);

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 10);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    rendererRef.current = renderer;

    // Add renderer to DOM
    containerRef.current.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 1);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Controls
    // const controls = new OrbitControls(camera, renderer.domElement);
    // controls.enableDamping = true;
    // controls.dampingFactor = 0.05;
    // controlsRef.current = controls;

    // Load GLB model
    const loader = new GLTFLoader();
    //"Ekana Stadium Low Poly Lucknow City game asset" (https://skfb.ly/oGYHn) by Shayan is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
    loader.load(
      "/3d-model/model.glb", // Replace with your actual stadium model path
      (gltf) => {
        const model = gltf.scene;
        model.scale.set(0.2, 0.2, 0.2); // Adjust scale as needed
        model.position.set(0, -5, 0);
        model.receiveShadow = true;
        model.castShadow = true;
        modelRef.current = model;
        scene.add(model);
      },
      (progress) => {
        let total = 3762932; //three.js total doesnt work with gltf files so we have to set it manually
        console.log("Loading progress:", (progress.loaded / total) * 100, "%");

        if (progress.loaded == total) {
          setModelLoaded(true);
          console.log("Model Loaded Successfully!");
        }
      },
      (error) => {
        console.error("Error loading model:", error);
      }
    );

    // Handle window resize
    const handleResize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;

      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    window.addEventListener("resize", handleResize);
    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      containerRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);
  useEffect(() => {
    let animationFrameId;

    const rotationAnimation = () => {
      animationFrameId = requestAnimationFrame(rotationAnimation);
      if (modelRef.current) {
        // console.log(rotationSpeed);
        modelRef.current.rotation.y += rotationSpeed * 0.001;
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };

    rotationAnimation();

    // Cleanup
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [rotationSpeed]); // Add rotationSpeed as dependency
  // Function to animate scale change
  const animateCamera = (
    newScale,
    newPosition,
    newRotation,
    duration = 1.5
  ) => {
    if (modelRef.current) {
      // Kill any existing animation
      if (animationRef.current) {
        animationRef.current.kill();
      }

      const timeline = gsap.timeline();

      // Animate scale and position simultaneously
      timeline.to(
        modelRef.current.scale,
        {
          x: newScale,
          y: newScale,
          z: newScale,
          duration: duration,
          ease: "power2.inOut",
          onUpdate: () => {
            setScale(modelRef.current.scale.x);
          },
        },
        0
      ); // The "0" means start at the beginning of timeline

      timeline.to(
        modelRef.current.position,
        {
          x: newPosition.x,
          y: newPosition.y,
          z: newPosition.z,
          duration: duration,
          ease: "power2.inOut",
          onUpdate: () => {
            setPosition({
              x: modelRef.current.position.x,
              y: modelRef.current.position.y,
              z: modelRef.current.position.z,
            });
          },
        },
        0
      ); // The "0" means start at the same time as scale
      timeline.to(
        modelRef.current.rotation,
        {
          x: newRotation.x,
          y: newRotation.y,
          z: newRotation.z,
          duration: duration,
          ease: "power2.inOut",
          onUpdate: () => {
            setRotation({
              x: modelRef.current.rotation.x,
              y: modelRef.current.rotation.y,
              z: modelRef.current.rotation.z,
            });
          },
        },
        0
      ); // The "0" means start at the same time as scale

      animationRef.current = timeline;
    }
  };
  const changeColor = (newColor) => {
    const targetColor = new THREE.Color(newColor);
    console.log("changing sky colour");
    gsap.to(colorRef.current, {
      r: targetColor.r,
      g: targetColor.g,
      b: targetColor.b,
      duration: 1,
      ease: "power2.inOut",
      onUpdate: () => {
        sceneRef.current.background = colorRef.current;
        setBackgroundColor(colorRef.current.getHexString());
      },
    });
  };

  // Effect to handle scale changes
  useEffect(() => {
    animateCamera(initialScale, { x: 0, y: -5, z: 0 }, { x: 0, y: 0, z: 0 });
  }, [initialScale]);

  useEffect(() => {
    if (modelRef.current) {
      modelRef.current.scale.set(scale, scale, scale);
    }
  }, [scale]);

  const showPlayers = () => {
    setRotationSpeed(1);
    setStep(2);
    changeColor("#80b9ff");
    animateCamera(0.5, { x: 0, y: -5, z: 0 }, { x: 0, y: 0, z: 0 });
  };
  const predictPlayers = () => {
    if (!teamA || !teamB || selectedA.length < 11 || selectedB.length < 11) {
      alert("Please select 11 players from each team to continue.");
      return;
    }

    if (!matchType || !matchDate) {
      alert("Please select match type and date.");
      return;
    }

    setRotationSpeed(20);
    setStep(1);
    changeColor("#97f6ff");

    animateCamera(
      0.5,
      { x: 0, y: 0, z: -40 },
      { x: Math.PI / 2, y: Math.PI / 2, z: 0 }
    );

    console.log(matchDate, matchType, selectedA, selectedB);
    setTimeout(() => {
      if (teamPredicted) {
        showPlayers();
      } else {
        setTimeout(() => {
          showPlayers();
        }, 3000);
      }
    }, 10000);
    // prediction logic here
    axios
      .post("/api/run_inference/", {
        team_1: teamA,
        team_2: teamB,
        team_1_players: selectedA,
        team_2_players: selectedB,
        match_date: matchDate,
        match_type: matchType,
      })
      .then((res) => {
        setTeamPredicted(true);

        console.log(res.data);
        setPrediction(res.data);
        axios.post("/api/formulate_reasons/", res.data).then((res) => {
          console.log(res.data);
          setPrediction(res.data);
          showPlayers();
        });
      });
  };
  const includeSuggestion = async (suggestion) => {
    setTeamA(suggestion.team1Name);
    setTeamB(suggestion.team2Name);
    setSelectedA([]);
    setSelectedB([]);

    setLoading(true);
    let playersTeam1 = await axios.post("/api/get_players", {
      team_name: suggestion.team1Name,
    });

    for (let i = 0; i < playersTeam1.data.players.length; i++) {
      if (suggestion.team1Players.includes(playersTeam1.data.players[i].id)) {
        setSelectedA((currentPlayers) => [
          ...currentPlayers,
          playersTeam1.data.players[i],
        ]);
      }
    }

    let playersTeam2 = await axios.post("/api/get_players", {
      team_name: suggestion.team2Name,
    });

    for (let i = 0; i < playersTeam2.data.players.length; i++) {
      if (suggestion.team2Players.includes(playersTeam2.data.players[i].id)) {
        setSelectedB((currentPlayers) => [
          ...currentPlayers,
          playersTeam2.data.players[i],
        ]);
      }
    }

    setLoading(false);
  };

  return (
    <div>
      <Loader isVisible={loading}>Autofilling Playing 11 for both teams</Loader>
      <div
        className="fixed left-0 top-0 z-0"
        ref={containerRef}
        style={{ width: "100vw", height: "100vh" }}
      />

      <TeamModal
        isVisible={teamModalVisibleA}
        setIsVisible={setTeamsModalVisibleA}
        selectedTeam={teamA}
        otherTeam={teamB}
        setSelectedTeam={setTeamA}
        setPlayersModalVisible={setPlayersModalVisibleA}
        setCurrentTeam={setCurrentTeam}
      />
      <TeamModal
        isVisible={teamModalVisibleB}
        setIsVisible={setTeamsModalVisibleB}
        selectedTeam={teamB}
        otherTeam={teamA}
        setSelectedTeam={setTeamB}
        setPlayersModalVisible={setPlayersModalVisibleB}
        setCurrentTeam={setCurrentTeam}
      />

      <PlayersModal
        isVisible={playersModalVisibleA}
        setIsVisible={setPlayersModalVisibleA}
        selectedPlayers={selectedA}
        setSelectedPlayers={setSelectedA}
        currentTeam={currentTeam}
      />

      <PlayersModal
        isVisible={playersModalVisibleB}
        setIsVisible={setPlayersModalVisibleB}
        selectedPlayers={selectedB}
        setSelectedPlayers={setSelectedB}
        currentTeam={currentTeam}
      />

      <div
        className="w-full h-full fixed left-0 transition-all overflow-auto"
        style={{
          transitionDuration: "0.5s",
          top: step == 0 ? "0" : "-100%",
        }}
      >
        <div className="text-xl md:text-2xl font-thin absolute z-10 left-1/2 top-0 -translate-x-1/2 p-4 md:p-5 md:px-10 bg-[#db4b4b] text-[#ffffff] rounded-b-lg text-center">
          Select Match Details
        </div>
        <div className="p-10 px-20 flex flex-col md:flex-row justify-evenly items-center flex-wrap w-full absolute top-20 left-0 z-100 gap-5 ">
          <div
            className="flex justify-between items-center p-5 text-center text-md md:text-2xl bg-[#ffffff] px-12 md:px-16 cursor-pointer"
            style={{
              clipPath: "polygon(15% 0%, 100% 0%, 85% 100%, 0% 100%)",
            }}
            onClick={() => setTeamsModalVisibleA(true)}
          >
            <span>
              {teamA || "Select Team"}
              {teamA ? " (" + selectedA.length + ")" : ""}
            </span>
            {teamA == "" && <PlusCircle size={20} className="ml-5" />}
          </div>
          <div className="flex flex-wrap flex-col gap-5">
            <select
              name="matchType"
              id="matchType"
              className="p-2 px-10 text-sm md:text-lg bg-transparent text-[#000000] outline-none"
              onChange={(e) => setMatchType(e.target.value)}
              style={{
                border: "3px solid #ffffff00",
                borderTopColor: "white",
                borderBottomColor: "white",
              }}
              defaultValue={""}
            >
              <option value="" disabled>
                Match Type
              </option>
              {options.map((option, index) => {
                return (
                  <option
                    value={option}
                    className="p-2 px-10 text-sm md:text-lg bg-transparent text-[#000000]"
                    key={index}
                  >
                    {option}
                  </option>
                );
              })}
            </select>
            <input
              type="date"
              name="date"
              id="date"
              className="p-2 px-10 text-sm md:text-lg bg-transparent text-[#000000]"
              onChange={(e) => setMatchDate(e.target.value)}
              style={{
                border: "3px solid #ffffff00",
                borderTopColor: "white",
                borderBottomColor: "white",
              }}
            />
          </div>
          <div
            className="flex justify-between items-center p-5 text-center text-md md:text-2xl bg-[#ffffff] px-12 md:px-16 cursor-pointer"
            style={{
              clipPath: "polygon(15% 0%, 100% 0%, 85% 100%, 0% 100%)",
            }}
            onClick={() => setTeamsModalVisibleB(true)}
          >
            <span>
              {teamB || "Select Team"}{" "}
              {teamB ? " (" + selectedB.length + ")" : ""}
            </span>
            {teamB == "" && <PlusCircle size={20} className="ml-5" />}
          </div>
          <div className="w-full flex justify-center items-center mt-10 ">
            <button
              className="bg-[#ffffff] md:bg-[#db4b4b] p-4 px-20 text-black  md:text-white"
              style={{
                clipPath:
                  "polygon(15% 0%, 85% 0%, 100% 50%, 85% 100%, 15% 100%, 0% 50%)",
              }}
              onClick={predictPlayers}
            >
              Next
            </button>
          </div>
          {/* <div className="mt-5 w-full text-center text-2xl">Suggested</div> */}
          <div className="mt-10 flex flex-wrap justify-center items-center gap-10">
            {suggestedMatches?.matches?.slice(0, 4).map((suggestion, index) => {
              return (
                <SuggestionCard
                  key={index}
                  matchDate={suggestedMatches.matchDate}
                  matchType={suggestedMatches.matchType}
                  teamA={suggestion.team1Name}
                  teamB={suggestion.team2Name}
                  onClick={() => {
                    includeSuggestion(suggestion);
                  }}
                ></SuggestionCard>
              );
            })}
          </div>
        </div>
      </div>
      {/* /loading part */}
      <div
        className="loading fixed left-1/2 -translate-x-1/2 text-2xl text-[#000000] z-10"
        style={{
          transition: "0.5s",
          transitionDelay: step == 1 ? "1s" : "0s",
          bottom: step == 1 ? "5rem" : "-4rem",
        }}
      >
        <span>Predicting Dream Team...</span>
      </div>
      {/* After Prediction */}
      <div
        className="absolute w-full h-full z-10 left-0 top-0 flex flex-wrap justify-start items-center"
        style={{
          transition: "0.5s",
          transitionDelay: step == 2 ? "1s" : "0s",
          opacity: step == 2 ? "1" : "0",
          scale: step == 2 ? "1" : "0.8",
          pointerEvents: step == 2 ? "all" : "none",
        }}
      >
        <PredictedView
          players={prediction?.players}
          matchDate={matchDate}
          matchType={matchType}
          teamA={teamA}
          teamB={teamB}
          overall_mae={prediction?.overall_mae}
          num_common={prediction?.num_common}
        ></PredictedView>
      </div>
    </div>
  );
};

export default StadiumViewer;
