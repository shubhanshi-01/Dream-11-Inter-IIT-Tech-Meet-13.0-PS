export async function translateText(text, targetLanguage, apiKey) {
  try {
    const url = `https://translation.googleapis.com/language/translate/v2?key=${apiKey}`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        q: text,
        target: targetLanguage,
        source: "en",
      }),
    });

    if (!response.ok) {
      throw new Error(`Translation failed: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      translation: data.data.translations[0].translatedText,
    };
  } catch (error) {
    return {
      translation: "",
      error: error instanceof Error ? error.message : "Translation failed",
    };
  }
}
