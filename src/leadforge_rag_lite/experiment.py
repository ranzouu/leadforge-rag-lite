from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import httpx

POLICIES_PATH = Path(__file__).resolve().parents[2] / "data" / "policies.json"
QUERIES_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_queries.json"


@dataclass
class Policy:
    id: str
    niche: str
    title: str
    content: str


@dataclass
class Query:
    id: str
    niche: str
    query: str
    expected_policy_id: str


@dataclass
class RunRecord:
    query_id: str
    niche: str
    query: str
    expected_policy_id: str
    retrieved_1: str = ""
    retrieved_2: str = ""
    retrieved_3: str = ""
    hit_at_3: bool = False
    mrr_at_3: float = 0.0
    latency_ms: float = 0.0
    answer: str = ""
    raw: str = ""


def _load_policies() -> list[Policy]:
    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        return [Policy(**item) for item in json.load(f)]


def _load_queries() -> list[Query]:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return [Query(**item) for item in json.load(f)]


def _tfidf_score(query: str, text: str) -> float:
    q_terms = [t.lower() for t in query.split() if len(t) > 3]
    text_lower = text.lower()
    if not q_terms:
        return 0.0
    score = 0.0
    for term in q_terms:
        if term in text_lower:
            score += 1.0
    return score / max(len(q_terms), 1)


def retrieve(policies: list[Policy], query: str, top_k: int = 3) -> list[tuple[Policy, float]]:
    scored = [(p, _tfidf_score(query, p.content)) for p in policies]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _call_llm(model: str, prompt: str) -> tuple[str, dict]:
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    timeout = httpx.Timeout(30.0)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Responda de forma objetiva, em português."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=timeout) as client:
        start = time.perf_counter()
        resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        latency_ms = (time.perf_counter() - start) * 1000

    if resp.status_code >= 400:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    return text, {"latency_ms": latency_ms}


_DEMO_ANSWERS = {
    "P1": "Para academias, priorizamos leads com foco em musculação e funcional, usando abertura por dor como organização de horários e retenção de alunos.",
    "P2": "Para petshops, priorizamos leads com atendimento presencial e banho/tosa, destacando redução de faltas e agenda automatizada.",
    "P3": "Para restaurantes, priorizamos delivery e atendimento local, usando abertura por cardápio digital e cupons.",
}


def _call_llm_demo(model: str, prompt: str, expected_policy_id: str) -> tuple[str, dict]:
    answer = _DEMO_ANSWERS.get(expected_policy_id, "Política aplicável não encontrada.")
    return answer, {"latency_ms": 110.0}


def run_experiment(models: Iterable[str], runs_dir: str = "runs") -> list[RunRecord]:
    policies = _load_policies()
    queries = _load_queries()
    runs: list[RunRecord] = []
    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    for model in models:
        for q in queries:
            start = time.perf_counter()
            top3 = retrieve(policies, q.query, top_k=3)
            latency_ms = (time.perf_counter() - start) * 1000

            retrieved_ids = [p.id for p, _ in top3]
            hit_at_3 = q.expected_policy_id in retrieved_ids
            mrr_at_3 = (1.0 / (retrieved_ids.index(q.expected_policy_id) + 1)) if hit_at_3 else 0.0

            context = "\n".join([p.content for p, _ in top3])
            prompt = f"Pergunta: {q.query}\nPolíticas:\n{context}\nResponda objetivamente."

            try:
                if os.getenv("LLM_API_KEY"):
                    answer, meta = _call_llm(model=model, prompt=prompt)
                    latency_ms = meta["latency_ms"]
                else:
                    answer, meta = _call_llm_demo(model=model, prompt=prompt, expected_policy_id=q.expected_policy_id)
            except Exception as exc:
                answer = ""
                latency_ms = -1.0

            runs.append(
                RunRecord(
                    query_id=q.id,
                    niche=q.niche,
                    query=q.query,
                    expected_policy_id=q.expected_policy_id,
                    retrieved_1=retrieved_ids[0] if len(retrieved_ids) > 0 else "",
                    retrieved_2=retrieved_ids[1] if len(retrieved_ids) > 1 else "",
                    retrieved_3=retrieved_ids[2] if len(retrieved_ids) > 2 else "",
                    hit_at_3=hit_at_3,
                    mrr_at_3=round(mrr_at_3, 4),
                    latency_ms=round(latency_ms, 2),
                    answer=answer,
                )
            )
    return runs


def save_runs(runs: list[RunRecord], runs_dir: str = "runs") -> tuple[Path, Path]:
    latest_path = Path(runs_dir) / "latest.csv"
    summary_path = Path(runs_dir) / "summary.csv"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(latest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RunRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in runs:
            writer.writerow(asdict(r))

    mrr_values = [r.mrr_at_3 for r in runs if r.mrr_at_3 > 0]
    summary = {
        "queries": len(runs),
        "hit_at_3": sum(1 for r in runs if r.hit_at_3),
        "mrr_at_3": round(sum(mrr_values) / len(mrr_values), 4) if mrr_values else 0.0,
        "avg_latency_ms": round(sum(r.latency_ms for r in runs if r.latency_ms > 0) / max(len(runs), 1), 2),
    }

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return latest_path, summary_path
