# backend/api.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models import call_llama_big, call_llama_fast, call_gpt_oss
from src.rag import load_knowledge_base, load_recipes
from src.knowledge_graphs import load_substitutions, build_fkg, build_skg
from src.experiment_runner import run_single_case
from src.evaluator import parse_llm_response

app = FastAPI(title="R.E.C.I.P.E. API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load data once at startup ──────────────────────────
print("Loading data...")
kb      = load_knowledge_base()
recipes = load_recipes()
fkg     = build_fkg(recipes)
skg     = build_skg(load_substitutions())
print(f"Ready! Loaded {len(recipes)} recipes and {len(kb)} science facts.")


class DiagnoseRequest(BaseModel):
    problem:   str
    recipe:    str
    condition: str = "full_system"


@app.get("/health")
def health():
    return {"status": "ok", "message": "R.E.C.I.P.E. API is running"}


@app.post("/diagnose")
def diagnose(request: DiagnoseRequest):
    test_case = {
        "Problem":  request.problem,
        "Recipe":   request.recipe,
        "Cause":    "",
        "Solution": "",
        "id":       0
    }

    # Run all 3 models
    result_big  = run_single_case(
        test_case, request.condition,
        call_llama_big, fkg, skg, kb, recipes)

    result_fast = run_single_case(
        test_case, request.condition,
        call_llama_fast, fkg, skg, kb, recipes)

    result_gpt  = run_single_case(
        test_case, request.condition,
        call_gpt_oss, fkg, skg, kb, recipes)

    # Parse responses
    result_big["parsed_response"]  = parse_llm_response(
        result_big.get("raw_response",  ""))
    result_fast["parsed_response"] = parse_llm_response(
        result_fast.get("raw_response", ""))
    result_gpt["parsed_response"]  = parse_llm_response(
        result_gpt.get("raw_response",  ""))

    return {
        "llama_big":  result_big,
        "llama_fast": result_fast,
        "gpt_oss":    result_gpt
    }