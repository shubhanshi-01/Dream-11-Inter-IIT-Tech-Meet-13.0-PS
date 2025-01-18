export async function textToSpeech(text, languageCode = "en-US") {
  try {
    const apiKey = process.env.NEXT_PUBLIC_GOOGLE_TTS_API_KEY;
    const apiEndpoint = `https://texttospeech.googleapis.com/v1/text:synthesize?key=${apiKey}`;
    console.log(languageCode);
    const requestBody = {
      input: { text },
      voice: {
        languageCode,
        ssmlGender: "MALE",
      },
      audioConfig: {
        audioEncoding: "MP3",
        pitch: 0,
        speakingRate: 1,
      },
    };

    const response = await fetch(apiEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Convert base64 to audio
    const audioContent = data.audioContent;
    const binaryAudio = atob(audioContent);
    const byteArray = new Uint8Array(binaryAudio.length);

    for (let i = 0; i < binaryAudio.length; i++) {
      byteArray[i] = binaryAudio.charCodeAt(i);
    }

    const audioBlob = new Blob([byteArray], { type: "audio/mp3" });
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

    return audio;
  } catch (error) {
    console.error("Text-to-speech error:", error);
    throw new Error("Failed to convert text to speech");
  }
}
