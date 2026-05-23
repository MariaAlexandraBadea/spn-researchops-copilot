from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.schema import DocumentChunk, RetrievedChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\-]+", text.lower())


class HybridRetriever:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.62,
    ):
        self.embedding_model_name = embedding_model
        self.alpha = alpha
        self.model: SentenceTransformer | None = None
        self.chunks: list[DocumentChunk] = []
        self.index = None
        self.bm25: BM25Okapi | None = None
        self.bm25_tokens: list[list[str]] = []
        self.embeddings: np.ndarray | None = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.embedding_model_name)
        return self.model

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        if not texts:
            raise ValueError("No chunks to index.")

        model = self._get_model()
        emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb.astype("float32")

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self.bm25_tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.bm25_tokens)

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise ValueError("Index not built.")
        faiss.write_index(self.index, str(index_dir / "faiss.index"))
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        with (index_dir / "bm25.pkl").open("wb") as f:
            pickle.dump({"tokens": self.bm25_tokens}, f)
        (index_dir / "meta.json").write_text(
            json.dumps({"embedding_model": self.embedding_model_name, "alpha": self.alpha}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> "HybridRetriever":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        obj = cls(embedding_model=meta["embedding_model"], alpha=meta.get("alpha", 0.62))
        obj.index = faiss.read_index(str(index_dir / "faiss.index"))
        obj.chunks = [
            DocumentChunk.from_dict(json.loads(line))
            for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with (index_dir / "bm25.pkl").open("rb") as f:
            data = pickle.load(f)
        obj.bm25_tokens = data["tokens"]
        obj.bm25 = BM25Okapi(obj.bm25_tokens)
        return obj

    def search(
        self,
        query: str,
        top_k: int = 6,
        source_type: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.index is None or self.bm25 is None:
            raise ValueError("Retriever is not built or loaded.")

        model = self._get_model()
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        k_search = min(max(top_k * 8, 20), len(self.chunks))
        v_scores, v_idx = self.index.search(q_emb, k_search)

        vector_map = {int(i): float(s) for i, s in zip(v_idx[0], v_scores[0]) if i >= 0}

        bm25_scores = self.bm25.get_scores(tokenize(query))
        if bm25_scores.max() > bm25_scores.min():
            bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
        else:
            bm25_norm = bm25_scores

        candidates = set(vector_map.keys())
        top_bm25_idx = np.argsort(-bm25_norm)[:k_search]
        candidates.update(int(i) for i in top_bm25_idx)

        results = []
        for i in candidates:
            chunk = self.chunks[i]
            if source_type and chunk.source_type != source_type:
                continue
            vs = vector_map.get(i, 0.0)
            bs = float(bm25_norm[i])
            score = self.alpha * vs + (1.0 - self.alpha) * bs

            # metadata boost for code-heavy queries
            q_low = query.lower()
            if any(tok in q_low for tok in ["function", "class", "code", "implementation", "method"]) and chunk.source_type == "code":
                score += 0.08
            if any(tok in q_low for tok in ["paper", "manuscript", "section", "latex"]) and chunk.source_type in {"paper", "latex"}:
                score += 0.05
            if any(tok in q_low for tok in ["metric", "rmse", "mae", "crps", "nll", "coverage"]) and chunk.source_type == "table":
                score += 0.08

            results.append(RetrievedChunk(chunk=chunk, score=float(score), vector_score=vs, bm25_score=bs))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
