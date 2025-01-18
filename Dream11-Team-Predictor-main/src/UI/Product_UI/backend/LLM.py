import os
from dotenv import load_dotenv
from openai import OpenAI
import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

from prompt import system_prompt
import random
import json

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def gemini8b(data):
    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 16000,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            enum=[],
            required=["reasons"],
            properties={
                "reasons": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.OBJECT,
                        enum=[],
                        required=["player_name", "player_id", "reason"],
                        properties={
                            "player_name": content.Schema(
                                type=content.Type.STRING,
                            ),
                            "player_id": content.Schema(
                                type=content.Type.STRING,
                            ),
                            "reason": content.Schema(
                                type=content.Type.STRING,
                            ),
                        },
                    ),
                ),
            },
        ),
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=system_prompt,
    )
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(json.dumps(data))

    return json.loads(response.text)


def get_reasons(data):

    response = gemini8b(data)

    # print(response)
    return response
