from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from warehouse_ai.core.events import WarehouseEvent
from warehouse_ai.core.schemas import CopilotDocument, DecisionAction, EvaluationMetric, ModelExplanation
from warehouse_ai.engines.base import BaseEngine, TrainResult

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:  # pragma: no cover - optional dependency
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


def chunk_text(text: str, chunk_size: int = 280) -> List[str]:
    words = text.split()
    return [
        " ".join(words[idx : idx + chunk_size])
        for idx in range(0, len(words), chunk_size)
    ] or [text]


class HybridVectorStore:
    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.counts = CountVectorizer(ngram_range=(1, 2), min_df=1)
        self.svd = TruncatedSVD(n_components=32, random_state=42)
        self.documents: List[Dict[str, str]] = []
        self.faiss_index = None
        self.dense_matrix: Optional[np.ndarray] = None
        self.term_matrix = None
        self.term_doc_lengths: Optional[np.ndarray] = None
        self.avg_doc_len = 1.0

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        self.documents = documents
        corpus = [doc["content"] for doc in documents]
        tfidf_matrix = self.tfidf.fit_transform(corpus)
        dense = self.svd.fit_transform(tfidf_matrix)
        dense = dense / np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-6)
        self.dense_matrix = dense.astype(np.float32)
        if faiss is not None:
            self.faiss_index = faiss.IndexFlatIP(self.dense_matrix.shape[1])
            self.faiss_index.add(self.dense_matrix)
        self.term_matrix = self.counts.fit_transform(corpus).astype(np.float32)
        self.term_doc_lengths = np.asarray(self.term_matrix.sum(axis=1)).squeeze(-1)
        self.avg_doc_len = float(np.mean(self.term_doc_lengths)) if len(self.documents) else 1.0

    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        if self.term_matrix is None:
            return np.zeros(len(self.documents), dtype=np.float32)
        query_vec = self.counts.transform([query])
        term_freq = self.term_matrix
        df = np.asarray((term_freq > 0).sum(axis=0)).ravel()
        idf = np.log((len(self.documents) - df + 0.5) / (df + 0.5) + 1)
        query_terms = query_vec.toarray().ravel() > 0
        scores = np.zeros(len(self.documents), dtype=np.float32)
        for term_idx in np.where(query_terms)[0]:
            tf = term_freq[:, term_idx].toarray().ravel()
            denom = tf + k1 * (1 - b + b * self.term_doc_lengths / max(self.avg_doc_len, 1e-6))
            scores += idf[term_idx] * ((tf * (k1 + 1)) / np.maximum(denom, 1e-6))
        return scores

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, str]]:
        if not self.documents:
            return []
        tfidf_query = self.tfidf.transform([text])
        dense_query = self.svd.transform(tfidf_query)
        dense_query = dense_query / np.maximum(np.linalg.norm(dense_query, axis=1, keepdims=True), 1e-6)
        if self.faiss_index is not None:
            _, indices = self.faiss_index.search(dense_query.astype(np.float32), top_k)
            dense_scores = np.zeros(len(self.documents), dtype=np.float32)
            for rank, doc_idx in enumerate(indices[0]):
                dense_scores[doc_idx] = 1.0 - rank * 0.05
        else:
            dense_scores = cosine_similarity(dense_query, self.dense_matrix)[0]
        sparse_scores = self._bm25(text)
        combined = 0.6 * dense_scores + 0.4 * (sparse_scores / np.maximum(np.max(sparse_scores), 1e-6))
        top_indices = np.argsort(-combined)[:top_k]
        return [{**self.documents[idx], "score": float(combined[idx])} for idx in top_indices]


@dataclass
class CopilotResponse:
    answer: str
    sources: List[str]
    action: Optional[DecisionAction]
    explanation: ModelExplanation


class WarehouseCopilotEngine(BaseEngine):
    engine_name = "copilot"

    def __init__(self, artifacts_dir: Path, api_key: Optional[str] = None) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.store = HybridVectorStore()
        self.documents: List[Dict[str, str]] = []
        self.events: List[WarehouseEvent] = []
        self.api_key = api_key
        if self.api_key and genai:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash-latest")
        else:
            self.model = None
            
        # Auto-load existing knowledge base if available
        knowledge_path = self.artifacts_dir / f"{self.engine_name}_knowledge.pkl"
        if knowledge_path.exists():
            try:
                with knowledge_path.open("rb") as handle:
                    data = pickle.load(handle)
                    self.documents = data.get("documents", [])
                    self.events = data.get("events", [])
                    if self.documents:
                        self.store.add_documents(self.documents)
            except Exception as e:
                print(f"Failed to auto-load copilot knowledge: {e}", file=sys.stderr)

    def ingest(self, documents: pd.DataFrame, events: List[WarehouseEvent]) -> TrainResult:
        chunks: List[Dict[str, str]] = []
        for row in documents.to_dict(orient="records"):
            for chunk_idx, chunk in enumerate(chunk_text(str(row["content"]))):
                chunks.append(
                    {
                        "doc_id": f"{row['doc_id']}::{chunk_idx}",
                        "title": str(row["title"]),
                        "content": chunk,
                        "source": str(row["source"]),
                        "timestamp": str(row["timestamp"]),
                    }
                )
        for event in events:
            event_text = (
                f"{event.event_type.value} subject={event.subject_id} warehouse={event.warehouse_id} "
                f"source={event.source} payload={json.dumps(event.payload, sort_keys=True)}"
            )
            chunks.append(
                {
                    "doc_id": f"event::{event.event_id}",
                    "title": f"{event.event_type.value} for {event.subject_id}",
                    "content": event_text,
                    "source": event.source,
                    "timestamp": event.timestamp.isoformat(),
                }
            )
        self.documents = chunks
        self.events = events
        self.store.add_documents(chunks)
        artifact_path = self.artifacts_dir / f"{self.engine_name}_knowledge.pkl"
        with artifact_path.open("wb") as handle:
            pickle.dump({"documents": chunks, "events": events}, handle)
        return TrainResult(
            engine_name=self.engine_name,
            model_version="v1",
            artifact_path=artifact_path,
            metrics=[
                EvaluationMetric(name="knowledge_chunks", value=float(len(chunks))),
                EvaluationMetric(name="event_count", value=float(len(events))),
            ],
            metadata={"retrieval": "hybrid_dense_bm25"},
        )

    def _parse_intent(self, question: str) -> str:
        text = question.lower()
        if "why" in text and "reorder" in text:
            return "decision_explanation"
        if "trigger reorder" in text or "create reorder" in text:
            return "action_reorder"
        if "anomal" in text or "risk" in text or "alert" in text:
            return "proactive_risk"
        return "general_query"

    def _find_sku(self, text: str) -> Optional[str]:
        match = re.search(r"(sku-\d{4})", text.lower())
        return match.group(1).upper() if match else None

    def _build_action(self, sku_id: str) -> DecisionAction:
        explanation = ModelExplanation(
            engine=self.engine_name,
            summary="Copilot parsed a reorder intent and constructed an operational action request.",
            confidence=0.78,
            feature_contributions={"intent_match": 0.6, "sku_extraction": 0.4},
            evidence=[f"sku_id={sku_id}"],
        )
        return DecisionAction(
            action_type="trigger_reorder",
            entity_id=sku_id,
            recommended_value="create_reorder_ticket",
            confidence=0.78,
            explanation=explanation,
            created_at=pd.Timestamp.utcnow().to_pydatetime(),
        )

    def answer(self, question: str) -> CopilotResponse:
        if not self.documents:
            raise RuntimeError("ingest knowledge before querying the copilot")
        
        # Local fallback for simple greetings
        text = question.lower().strip()
        if text in ["hi", "hello", "hey", "hola"]:
            return CopilotResponse(
                answer="Hello! I'm the Warehouse Intelligence Copilot. How can I help you optimize your operations today?",
                sources=[],
                action=None,
                explanation=ModelExplanation(
                    engine=self.engine_name,
                    summary="Local greeting match.",
                    confidence=1.0,
                    feature_contributions={"greeting_match": 1.0},
                    evidence=[]
                )
            )

        intent = self._parse_intent(question)
        retrieved = self.store.query(question, top_k=5)
        sources = [f"{doc['source']}::{doc['doc_id']}" for doc in retrieved]
        answer = ""
        action = None
        if intent == "decision_explanation":
            sku_id = self._find_sku(question)
            evidence = [doc for doc in retrieved if (sku_id and sku_id in doc["content"]) or "reorder" in doc["content"]]
            if evidence:
                answer = (
                    f"Reorder activity for {sku_id or 'the requested SKU'} was driven by the demand and inventory evidence below: "
                    + " ".join(doc["content"] for doc in evidence[:2])
                )
            else:
                answer = "No explicit reorder event was found, so the copilot is returning the closest operational evidence from demand and inventory history."
        elif intent == "action_reorder":
            sku_id = self._find_sku(question) or "SKU-0000"
            action = self._build_action(sku_id)
            answer = f"Prepared a reorder action for {sku_id}. Review the action payload before execution."
        elif intent == "proactive_risk":
            risks = [doc for doc in retrieved if any(keyword in doc["content"] for keyword in ["anomaly", "maintenance", "inventory"])]
            answer = "Top risk signals: " + " ".join(doc["title"] for doc in risks[:3] or retrieved[:3])
        else:
            if self.model:
                context = "\n---\n".join([doc["content"] for doc in retrieved])
                prompt = (
                    f"You are the Warehouse Intelligence Copilot. Answer the user's question using the provided context from warehouse events and documents.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Concise Answer:"
                )
                try:
                    response = self.model.generate_content(prompt)
                    answer = response.text.strip()
                except Exception as e:
                    print(f"Gemini error: {e}", file=sys.stderr)
                    answer = f"Gemini error (falling back to retrieval): {' '.join(doc['content'] for doc in retrieved[:3])}"
            else:
                answer = " ".join(doc["content"] for doc in retrieved[:3])
        
        if not answer:
             answer = "I've analyzed the warehouse events but couldn't find a specific answer for that. Could you rephrase or ask about current inventory/anomalies?"
             
        return CopilotResponse(
            answer=answer,
            sources=sources,
            action=action,
            explanation=ModelExplanation(
                engine=self.engine_name,
                summary="Response synthesized from hybrid dense and sparse retrieval over documents, events, and decision history.",
                confidence=float(np.clip(0.55 + len(retrieved) * 0.05, 0.3, 0.95)),
                feature_contributions={
                    "retrieved_chunks": float(len(retrieved)),
                    "intent_score": 0.8 if intent != "general_query" else 0.5,
                },
                evidence=sources[:3],
            ),
        )
