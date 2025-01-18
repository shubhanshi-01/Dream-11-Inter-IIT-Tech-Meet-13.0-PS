import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const IndianLanguageSelect = ({ onChange, language }) => {
  console.log(language);
  const languages = [
    { code: "en", name: "English" },
    // { code: "as", name: "অসমীয়া" },
    { code: "bn", name: "বাংলা" },
    { code: "gu", name: "ગુજરાતી" },
    { code: "hi", name: "हिन्दी" },
    { code: "kn", name: "ಕನ್ನಡ" },
    { code: "ml", name: "മലയാളം" },
    { code: "mr", name: "मराठी" },
    // { code: "ne", name: "नेपाली" },
    // { code: "or", name: "ଓଡ଼ିଆ" },
    { code: "pa", name: "ਪੰਜਾਬੀ" },
    // { code: "sd", name: "سنڌي" },
    { code: "ta", name: "தமிழ்" },
    { code: "te", name: "తెలుగు" },
    { code: "ur", name: "اردو" },
  ];

  return (
    <Select
      onValueChange={onChange}
      className="outline-none border-none bg-zinc-300 color-white"
      defaultValue={language}
    >
      <SelectTrigger className="w-32 outline-none border-none bg-zinc-300 color-white">
        <SelectValue placeholder="Select a language" />
      </SelectTrigger>
      <SelectContent className="outline-none border-none bg-zinc-300 color-white">
        {languages.map((lang) => (
          <SelectItem key={lang.code} value={lang.code}>
            {lang.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default IndianLanguageSelect;
