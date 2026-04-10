# src/models.py

import os
from dotenv import load_dotenv
from groq import Groq

# -------------------- LOAD ENV --------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- GROQ CLIENT --------------------
groq_client = Groq(api_key=GROQ_API_KEY)


# -------------------- GROQ MODELS --------------------
# -------------------- GROQ MODELS --------------------

def call_llama_big(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9
    )
    return response.choices[0].message.content


def call_llama_fast(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9
    )
    return response.choices[0].message.content


# -------------------- GEMINI (COMMENTED OUT) --------------------
"""
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def call_gemini(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text
"""


# -------------------- TEST FUNCTION --------------------
def test_all_models():
    prompt = "In one sentence, what causes hollandaise sauce to break?"

    models = {
    "LLaMA 3.1 70B": call_llama_big,
    "LLaMA 3.1 8B": call_llama_fast
        # "Gemini": call_gemini
    }

    print("Testing all models...")
    print("-" * 50)

    for name, fn in models.items():
        try:
            response = fn(prompt)
            print(f"OK  {name}: {response[:80]}...")
        except Exception as e:
            print(f"ERR {name}: {e}")

    print("-" * 50)
    print("Done!")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    test_all_models()