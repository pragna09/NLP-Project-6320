# src/models.py
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
#groq_client = Groq(api_key=GROQ_API_KEY)


def call_gemini_flash(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(prompt).text


def call_gemini_pro(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-pro")
    return model.generate_content(prompt).text


#def call_llama(prompt: str) -> str:
#    r = groq_client.chat.completions.create(
#        model="llama-3.1-70b-versatile",
#        messages=[{"role": "user", "content": prompt}],
#        temperature=0.7,
#        max_tokens=1024,
#        top_p=0.9
#    )
#    return r.choices[0].message.content



#def call_mixtral(prompt: str) -> str:
#    r = groq_client.chat.completions.create(
#        model="mixtral-8x7b-32768",
#        messages=[{"role": "user", "content": prompt}],
#        temperature=0.7,
#        max_tokens=1024,
#        top_p=0.9
#    )
#    return r.choices[0].message.content


def test_all_models():
    prompt = "In one sentence, what causes hollandaise sauce to break?"
    models = {
        "Gemini Flash":  call_gemini_flash,
        "Gemini Pro":    call_gemini_pro,
         #"Llama 3.1 70B": call_llama,
         #"Mixtral 8x7B":  call_mixtral
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


if __name__ == "__main__":
    test_all_models()