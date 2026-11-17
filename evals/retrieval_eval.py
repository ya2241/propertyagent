"""Compare plain keyword filtering with vector search on a small hand-labelled query set.

Run:  python -m evals.retrieval_eval
Metrics: precision@3, hit-rate@3 (any relevant result in the top 3) and MRR.
"""
from __future__ import annotations

from app.data import PROPERTIES
from app.retrieval import KeywordRetriever, VectorRetriever

LABELLED = [
    ("dog friendly home", {"P002", "P013", "P017", "P009"}),
    ("quiet place to work remotely", {"P001", "P007", "P005", "P020"}),
    ("good for kids and schools", {"P002", "P004", "P013"}),
    ("cheap option for a student", {"P006", "P016", "P019"}),
    ("lively area with bars and cafes", {"P012", "P015", "P011"}),
    ("easy commute to the financial district", {"P014", "P003", "P010"}),
    ("wheelchair accessible", {"P017"}),
    ("somewhere to park my car", {"P004", "P005", "P013", "P017", "P018"}),
    ("low energy bills and sustainable", {"P020"}),
    ("pets welcome", {"P002", "P013", "P017", "P009"}),
    ("gym and concierge", {"P003", "P005", "P010", "P014", "P018"}),
    ("outdoor space", {"P002", "P004", "P009", "P011", "P013", "P017", "P018", "P020", "P010"}),
]


def evaluate(retriever, k: int = 3) -> dict:
    ids = [p.id for p in PROPERTIES]
    prec, hits, rr = [], [], []
    for query, relevant in LABELLED:
        top = [pid for pid, _ in retriever.rank(query, ids, k)]
        found = [pid in relevant for pid in top]
        prec.append(sum(found) / k)
        hits.append(1.0 if any(found) else 0.0)
        rr.append(next((1 / (i + 1) for i, f in enumerate(found) if f), 0.0))
    n = len(LABELLED)
    return {"precision@3": sum(prec) / n, "hit_rate@3": sum(hits) / n, "mrr": sum(rr) / n}


def main() -> None:
    results = {"keyword": evaluate(KeywordRetriever(PROPERTIES)), "vector": evaluate(VectorRetriever(PROPERTIES))}
    print(f"{'retriever':<10} {'P@3':>6} {'hit@3':>7} {'MRR':>6}   ({len(LABELLED)} queries)")
    for name, m in results.items():
        print(f"{name:<10} {m['precision@3']:>6.2f} {m['hit_rate@3']:>7.2f} {m['mrr']:>6.2f}")


if __name__ == "__main__":
    main()
