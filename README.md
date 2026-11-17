# PropertyAgent

A tool-calling LLM agent behind a FastAPI service that qualifies lettings enquiries. Built with Python, FastAPI, the Claude API, Pydantic and Pytest.

- Extracts applicant preferences (budget, bedrooms, move-in date, area) into Pydantic-validated structures
- Tools: property search, viewing-slot lookup, viewing booking, human handover
- **Confirmation gate**: write actions (`book_viewing`) never run until the client sends `confirm: true`
- **Handover** when information is missing, extraction confidence is low, a tool keeps failing, nothing matches, or the user asks for a person
- Eval suite of 14 conversations (success, missing info, failures) with pass-rate tracking per prompt/context version
- Retrieval over property descriptions: vector search vs plain keyword filtering, with a labelled comparison
- Per-turn and per-conversation latency, token and cost logging

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...                # real model
uvicorn app.main:app --reload
# or, no key needed (rule-based stand-in for the model):
PROPERTYAGENT_LLM=mock uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "I need a 2 bed under £2400 from 2026-12-20"}'
# ... later, to confirm a pending booking:
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"conversation_id": "<id>", "message": "Yes", "confirm": true}'
```

`status` is `ok`, `awaiting_confirmation` (see `pending_action`) or `handover` (see `handover_reason`).
Settings: `PROPERTYAGENT_MODEL`, `PROMPT_VERSION` (v1/v2/v3), `RETRIEVAL_MODE` (vector/keyword), `PRICE_INPUT_PER_MTOK` / `PRICE_OUTPUT_PER_MTOK` (set to your model's real prices; defaults are placeholders).

## Layout

```
app/agent.py      agent loop, confirmation gate, handover rules, grounding check
app/tools.py      tool registry; arguments validated with Pydantic before execution
app/schemas.py    preferences, tool args, API models
app/retrieval.py  keyword vs vector retrievers
app/telemetry.py  latency / tokens / cost -> logs/conversations.jsonl
app/prompts.py    prompt versions v1 -> v3
evals/            conversation evals, retrieval eval
tests/            31+ pytest tests
```

## Tests and evals

```bash
pytest
python -m evals.retrieval_eval
python -m evals.run_evals --mode mock                       # offline harness check
python -m evals.run_evals --mode live --prompt-version v1   # real Claude; run v1, v2, v3 then:
python -m evals.run_evals --compare                         # pass-rate history across iterations
```

**Important:** `--mode mock` uses a deterministic rule-based stand-in so the pipeline runs without a key. Its 14/14 only shows the harness, gating and handover logic work; it says nothing about Claude. Record real pass rates with `--mode live` and put them in the table below.

| run | prompt | context | pass rate | notes |
|-----|--------|---------|-----------|-------|
| _fill in from `--compare`_ | v1 | history only | | |
| | v2 | + grounding rules | | |
| | v3 | + workflow rules, recorded-state block | | |

## Retrieval: vector vs keyword

12 hand-labelled queries over 20 properties (`python -m evals.retrieval_eval`):

| retriever | precision@3 | hit-rate@3 | MRR |
|-----------|------------|-----------|-----|
| keyword   | 0.64 | 0.92 | 0.88 |
| vector    | 0.69 | 1.00 | 0.90 |

Caveats: the default embedder is a local LSA model (TF-IDF + SVD, numpy only) so everything runs offline; `EMBEDDER=st` swaps in sentence-transformers. The set is tiny and the LSA dimension (12) was picked by looking at this same set, so treat the gap as indicative. Vector search helps on paraphrased queries ("good for kids and schools"); keyword wins on exact terms and is easier to explain.

## Failure modes and mitigations

| failure | mitigation | trade-off |
|---------|-----------|-----------|
| Hallucinated property IDs / rents | `check_grounding` compares every `P###` and `£` amount in a reply with tool results and user messages; one corrective retry, then handover | extra LLM call on a miss; false positives on figures such as deposits |
| Tool errors (bad args, unavailable, slot taken) | Pydantic-validated args; errors returned to the model as `is_error` results; handover after 2 errors | a flaky tool can hand over a recoverable conversation |
| Unwanted writes | `book_viewing` is held server-side until `confirm: true`; declined/ignored confirmations are fed back as errors | one extra round trip per booking |
| Guessed preferences | `confidence` on `record_preferences`; below 0.5 triggers handover | trusts the model's self-reported confidence |
| Endless clarifying | handover after 4 turns without budget and bedrooms | may hand over users who are just slow to answer |
| LLM outage | any exception becomes a logged handover | |

Add your own findings from live runs here (what failed, what you changed, what it cost).

## Known limits

In-memory conversations and bookings (swap `ConversationStore` / `PropertyStore` for a database), synthetic inventory, no authentication.
