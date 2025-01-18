import Image from "next/image";
import { useState } from "react";

export const PlayerImage = ({
  src,
  alt,
  size = 500,
  className,
  fallbackPadding = "20px",
}) => {
  const fallbackImage = "/fallback.png";
  const [error, setError] = useState(false);

  return (
    <Image
      className={className}
      src={!error ? src : fallbackImage}
      alt={alt}
      width={size}
      height={size}
      style={error ? { padding: fallbackPadding } : {}}
      onError={() => setError(true)}
    />
  );
};
