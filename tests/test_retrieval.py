from app.data import PROPERTIES
from app.retrieval import KeywordRetriever, VectorRetriever
from evals.retrieval_eval import evaluate

IDS = [p.id for p in PROPERTIES]


def test_keyword_exact_match():
    top = [pid for pid, _ in KeywordRetriever(PROPERTIES).rank("wheelchair", IDS, 3)]
    assert top == ["P017"]


def test_vector_finds_pet_friendly_homes():
    top = {pid for pid, _ in VectorRetriever(PROPERTIES).rank("dog friendly home", IDS, 3)}
    assert top & {"P002", "P013"}


def test_vector_respects_candidate_subset():
    out = VectorRetriever(PROPERTIES).rank("garden", ["P003", "P006"], 5)
    assert {pid for pid, _ in out} <= {"P003", "P006"}


def test_empty_query_returns_nothing():
    assert VectorRetriever(PROPERTIES).rank("the a of", IDS, 3) == []


def test_vector_at_least_matches_keyword_on_labelled_set():
    kw, vec = evaluate(KeywordRetriever(PROPERTIES)), evaluate(VectorRetriever(PROPERTIES))
    assert vec["hit_rate@3"] >= kw["hit_rate@3"]
