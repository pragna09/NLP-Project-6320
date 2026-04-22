# src/models.py

import os
from pathlib import Path 
from dotenv import load_dotenv
from groq import Groq

# -------------------- LOAD ENV --------------------
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- GROQ CLIENT --------------------
groq_client = Groq(api_key=GROQ_API_KEY)

# -------------------- GROQ MODEL: LLAMA 3.3 --------------------
def call_llama_big(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9
    )
    return response.choices[0].message.content


# -------------------- GROQ MODEL: LLAMA 3.1 --------------------
def call_llama_fast(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9
    )
    return response.choices[0].message.content


# -------------------- GROQ MODEL: GPT-OSS 20B --------------------
def call_gpt_oss(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a culinary expert and food scientist. "
                    "Always respond using EXACTLY these three labeled sections:\n"
                    "CAUSE: [what went wrong]\n"
                    "SOLUTION: [how to fix it]\n"
                    "EXPLANATION: [the science behind it]\n"
                    "Never deviate from this format. Never add extra sections."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9
    )
    return response.choices[0].message.content


# -------------------- TEST FUNCTION --------------------
def test_all_models():
    prompt = "In one sentence, what causes hollandaise sauce to break?"

    models = {
        "LLaMA 3.3 70B": call_llama_big,
        "LLaMA 3.1 8B": call_llama_fast,
        "GPT-OSS 20B": call_gpt_oss
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