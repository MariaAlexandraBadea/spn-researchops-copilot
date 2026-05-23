from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import route_message
from app.retrieval import HybridRetriever

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
METRICS_ROOT = os.getenv("METRICS_ROOT", "data/source_repo")

app = FastAPI(title="SPN ResearchOps Copilot API")
retriever: HybridRetriever | None = None


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup():
    global retriever
    retriever = HybridRetriever.load(INDEX_DIR)


@app.post("/ask")
def ask(req: AskRequest):
    if retriever is None:
        return {"error": "Retriever not loaded."}
    return route_message(req.question, retriever, metrics_root=METRICS_ROOT)


@app.get("/health")
def health():
    return {"status": "ok", "index_dir": INDEX_DIR}
