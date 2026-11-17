from app.config import Settings
from app.llm import RuleBasedLLM
from evals.cases import CASES
from evals.run_evals import run_case


def test_eval_harness_passes_with_rule_based_stand_in():
    failed = [(r["name"], r["failures"]) for r in (run_case(c, RuleBasedLLM(), Settings()) for c in CASES) if not r["passed"]]
    assert not failed, failed
