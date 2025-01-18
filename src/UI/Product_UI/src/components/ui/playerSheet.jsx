import { X, ChevronDown, ChevronUp, Speech } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetClose,
} from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { PlayerImage } from "@/components/ui/PlayerImage";
import { textToSpeech } from "@/lib/speechSynth.js";
import React, { useState, useEffect } from "react";
import PlayerCard from "@/components/ui/playerCard";
import LeftPanel from "@/components/ui/LeftPanel";
import { translateText } from "@/lib/translate.js";
import IndianLanguageSelect from "./langSelect";
import Image from "next/image";
import { Icon } from "lucide-react";
import { cricketWicket } from "@lucide/lab";

const colors = [
  "#ff585870",
  "#6bf4f85c",
  "#f8c26b5c",
  "#ff00ff38",
  "#5fa8ff73",
  "#00ff8c61",
  "#6200ff50",
  "#ff008963",
];

const StatItem = ({ label, value, color }) => (
  <div
    className="bg-white/5 p-4 rounded-lg"
    // style={
    //   color
    //     ? {
    //         background: color,
    //       }
    //     : {}
    // }
  >
    <p className="text-white/50 text-sm mb-1">{label}</p>
    <p className="text-white text-lg font-light">{value.toFixed(2)}</p>
  </div>
);

const FormItem = ({ match, runs, wickets, date }) => (
  <div className="flex justify-between items-center bg-white/5 p-3 rounded-lg">
    <div>
      <p className="text-white text-sm font-medium">vs {match}</p>
      <p className="text-white/70 text-xs">{date}</p>
    </div>
    <p className="text-white font-medium text-sm flex gap-1">
      <Image
        src="/cricket-bat.png"
        width="100"
        height="100"
        className="w-5 h-5 invert-[100%] opacity-50"
        alt="cricket bat"
      />
      {runs}
    </p>
    <p className="text-white font-medium text-sm flex gap-1">
      <Image
        src="/cricket-wicket.png"
        width="100"
        height="100"
        className="w-5 h-5 invert-[100%] opacity-50"
        alt="cricket bat"
      />
      {wickets}
    </p>
  </div>
);

const PlayerSheet = ({ player, open, onOpenChange }) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [language, setLanguage] = useState("en");
  const [audioRef, setAudioRef] = useState();

  const speak = async () => {
    let translated_text = player.reason;
    if (language != "en") {
      translated_text = await translateText(
        player.reason,
        language,
        process.env.NEXT_PUBLIC_GOOGLE_TRANSLATE_API_KEY
      );
      translated_text = translated_text.translation;
    }
    console.log(translated_text);
    if (audioRef) {
      audioRef.pause();
    }
    let audio = await textToSpeech(
      translated_text ||
        "This player is a top performer known for their consistent match-winning abilities, adaptability, and strong leadership skills.",
      language + "-IN"
    );
    setAudioRef(audio);
    audio.play();
  };

  useEffect(() => {
    if (open)
      setTimeout(() => {
        setIsDropdownOpen(true);
      }, 500);
  }, [open]);

  if (!player) return null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-[400px] bg-white/10 backdrop-blur-xl border-l border-white/20 p-0 [&>button]:hidden">
        <div className="absolute top-6 left-6 z-10 flex flex-wrap justify-start items-start gap-5">
          <div
            className="content-box p-3 bg-zinc-500 rounded-full cursor-pointer transition-all hover:scale-125 hover:bg-zinc-700 aspect-square"
            onClick={() => {
              speak();
            }}
          >
            <Speech size={20} className="text-white" />
            {/* <IndianLanguageSelect onChange={setLanguage} /> */}
          </div>
          <IndianLanguageSelect
            onChange={setLanguage}
            language={language}
          ></IndianLanguageSelect>
        </div>
        <div className="absolute top-4 right-4 z-50">
          <SheetClose asChild>
            <Button
              variant="ghost"
              size="icon"
              className="text-white hover:bg-white/10 hover:text-white"
            >
              <X size={28} strokeWidth={1.5} />
              <span className="sr-only">Close</span>
            </Button>
          </SheetClose>
        </div>

        <ScrollArea className="h-full w-full px-6 pb-6 pt-20">
          <SheetHeader className="mb-6">
            <SheetTitle className="text-2xl font-extralight text-white">
              {player.name}
            </SheetTitle>
            <SheetDescription className="text-white/70">
              {player.teamName}
            </SheetDescription>
          </SheetHeader>

          {/* Player Image */}
          <div className="relative w-full h-64 mb-6 rounded-xl overflow-hidden">
            <PlayerImage
              src={player.image}
              alt={player.name}
              className="w-full h-full object-contain"
            />
          </div>

          {/* Player Type Badge */}
          <div className="mb-6 flex gap-5">
            <span className="inline-block px-4 py-2 bg-[#ffc800] rounded-full text-[#795f00] font-medium">
              {player.player_type}
            </span>
            <span className="inline-block px-4 py-2 bg-[#6fc1ff] rounded-full text-[#143c5b] font-medium">
              {player.stats.player_role}
            </span>
          </div>

          <Separator className="my-6 bg-white/10" />

          <div className="mt-4 mb-4">
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex justify-between items-center w-full bg-[#00ffa2]/80 px-4 py-3 rounded-lg text-[#002e25]/80 font-medium hover:bg-[#00ffa2]/100"
            >
              Why this player?
              {isDropdownOpen ? (
                <ChevronUp size={20} className="text-black" />
              ) : (
                <ChevronDown size={20} className="text-black" />
              )}
            </button>
            <div
              className={`transition-[max-height,opacity] duration-500 ease-in-out overflow-hidden ${
                isDropdownOpen
                  ? "max-h-[300px] opacity-100"
                  : "max-h-0 opacity-0"
              }`}
              style={{
                transitionTimingFunction: "cubic-bezier(0.4, 0, 0.2, 1)", // Smooth curve
              }}
            >
              <div
                className="mt-3 bg-[#000]/15 text-sm text-[white]/80 p-4 rounded-lg"
                style={{
                  fontFamily: "Poppins",
                  lineHeight: "1.6",
                  letterSpacing: "0.5px",
                }}
              >
                <p>
                  {player.reason ||
                    "This player is a top performer known for their consistent match-winning abilities, adaptability, and strong leadership skills."}
                </p>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="space-y-6">
            <h3 className="text-white/90 text-xl font-medium">
              Performance Stats
            </h3>
            <div className="grid grid-cols-2 gap-6">
              <StatItem
                label="Matches"
                value={player.stats.matches}
                color={colors[0]}
              />
              {player.stats.matches > 0 && (
                <StatItem
                  label="Runs"
                  value={player.stats.runs}
                  color={colors[1]}
                />
              )}
              {player.stats.matches > 0 && (
                <StatItem
                  label="Batting Avg"
                  value={player.stats.batting_avg}
                  color={colors[2]}
                />
              )}
              {player.stats.strike_rate > 0 && (
                <StatItem
                  label="Strike Rate"
                  value={player.stats.strike_rate}
                  color={colors[3]}
                />
              )}
              {player.stats.centuries > 0 && (
                <StatItem
                  label="100s"
                  value={player.stats.centuries}
                  color={colors[4]}
                />
              )}
              {player.stats.half_centuries > 0 && (
                <StatItem
                  label="50s"
                  value={player.stats.half_centuries}
                  color={colors[5]}
                />
              )}
              {player.stats.bowling_avg != -1 && (
                <StatItem
                  label="Bowling Avg"
                  value={player.stats.bowling_avg}
                  color={colors[6]}
                />
              )}
              {player.stats.economy_rate != -1 && (
                <StatItem
                  label="Economy Rate"
                  value={player.stats.economy_rate}
                  color={colors[7]}
                />
              )}
            </div>
          </div>

          <Separator className="my-6 bg-white/10" />

          {/* Recent Form */}
          <div className="space-y-4 mb-6">
            <h3 className="text-white/90 text-xl font-medium">Recent Form</h3>
            <div
              className="space-y-2"
              style={{
                fontFamily: "Poppins",
              }}
            >
              {player.stats.recent_form.map((form, index) => (
                <FormItem
                  key={index}
                  match={form.match.opponent}
                  runs={form.runs}
                  wickets={form.wickets}
                  date={form.match.date}
                />
              ))}
            </div>
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
};

export default PlayerSheet;
