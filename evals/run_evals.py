"""Run the conversation eval suite and track pass rate across prompt/context iterations.

  python -m evals.run_evals --mode mock                       # offline harness check (rule-based stand-in)
  python -m evals.run_evals --mode live --prompt-version v3   # real Claude (needs ANTHROPIC_API_KEY)
  python -m evals.run_evals --compare                         # pass-rate history

Every run appends one line to evals/results/runs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from app.agent import Conversation, PropertyAgent, check_grounding
from app.config import Settings
from app.llm import AnthropicLLM, RuleBasedLLM
from app.tools import PropertyTools

from .cases import CASES, Case

RESULTS = Path(__file__).parent / "results" / "runs.jsonl"


def run_case(case: Case, llm, settings: Settings) -> dict:
    tools = PropertyTools(settings=settings, inject_failures=case.inject_failures)
    agent = PropertyAgent(llm, tools, settings)
    conv, replies, statuses = Conversation(), [], []
    for t in case.turns:
        r = agent.handle(conv, t.message, t.confirm)
        replies.append(r.reply)
        statuses.append(r.status)
    called = [b["name"] for m in conv.messages if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    e, fails = case.expect, []

    def check(ok, msg):
        if not ok:
            fails.append(msg)

    if "status" in e: check(statuses[-1] == e["status"], f"status {statuses[-1]} != {e['status']}")
    for n in e.get("tools_called", []): check(n in called, f"expected tool {n}")
    for n in e.get("tools_not_called", []): check(n not in called, f"unexpected tool {n}")
    for k, v in e.get("prefs", {}).items(): check(getattr(conv.prefs, k) == v, f"pref {k}={getattr(conv.prefs, k)}")
    if "bookings" in e: check(len(tools.bookings) == e["bookings"], f"bookings {len(tools.bookings)} != {e['bookings']}")
    if "tool_errors" in e: check(conv.tool_errors == e["tool_errors"], f"tool_errors {conv.tool_errors} != {e['tool_errors']}")
    for s in e.get("reply_has", []): check(s.lower() in replies[-1].lower(), f"reply missing '{s}'")
    # invariants checked for every case
    check(len(tools.bookings) <= sum(1 for t in case.turns if t.confirm is True), "booking without confirmation")
    ungrounded = [i for r_ in replies for i in check_grounding(r_, conv)]
    check(not ungrounded, f"ungrounded details in reply: {ungrounded}")

    m = conv.totals
    return {"name": case.name, "category": case.category, "passed": not fails, "failures": fails,
            "input_tokens": m.input_tokens, "output_tokens": m.output_tokens, "cost_usd": m.cost_usd,
            "latency_ms": m.latency_ms, "grounding_issues": m.grounding_issues}


def run_suite(mode: str, prompt_version: str, label: str | None) -> dict:
    settings = replace(Settings.from_env(), prompt_version=prompt_version)
    llm = RuleBasedLLM() if mode == "mock" else AnthropicLLM(settings.model)
    results = [run_case(c, llm, settings) for c in CASES]
    passed = sum(r["passed"] for r in results)
    return {"label": label or f"{mode}-{prompt_version}", "mode": mode, "prompt_version": prompt_version,
            "model": settings.model if mode == "live" else "rule-based", "retrieval": settings.retrieval_mode,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "passed": passed, "total": len(results),
            "pass_rate": round(passed / len(results), 3),
            "cost_usd": round(sum(r["cost_usd"] for r in results), 5),
            "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in results), "cases": results}


def compare() -> None:
    if not RESULTS.exists():
        print("No runs recorded yet.")
        return
    print(f"{'label':<24}{'mode':<6}{'prompt':<7}{'pass':>8}{'rate':>7}{'cost $':>9}")
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        print(f"{r['label']:<24}{r['mode']:<6}{r['prompt_version']:<7}{r['passed']:>3}/{r['total']:<4}{r['pass_rate']:>7.0%}{r['cost_usd']:>9.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--prompt-version", default="v3")
    ap.add_argument("--label")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    if a.compare:
        return compare()
    run = run_suite(a.mode, a.prompt_version, a.label)
    for r in run["cases"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  [{r['category']:<12}] {r['name']}" + ("" if r["passed"] else f"  -> {r['failures']}"))
    print(f"\n{run['passed']}/{run['total']} passed ({run['pass_rate']:.0%})  tokens={run['tokens']}  cost=${run['cost_usd']:.4f}")
    if not a.no_save:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS, "a") as f:
            f.write(json.dumps({k: v for k, v in run.items() if k != "cases"} |
                               {"cases": {c["name"]: c["passed"] for c in run["cases"]}}) + "\n")


if __name__ == "__main__":
    main()
