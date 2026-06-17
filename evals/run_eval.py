"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    q_text = question["question"]
    gold_sql = question["gold_sql"]

    gold_ok, gold_rows, gold_error = run_sql(db_id, gold_sql)

    started = time.monotonic()
    try:
        response = httpx.post(
            agent_url,
            json={
                "question": q_text,
                "db": db_id,
                "tags": {"run": "eval"},
            },
            timeout=120.0,
        )
        latency = time.monotonic() - started
        response.raise_for_status()
        agent_payload = response.json()
        agent_error = None
    except Exception as e:  # noqa: BLE001
        latency = time.monotonic() - started
        agent_payload = {}
        agent_error = f"{type(e).__name__}: {e}"

    attempts: list[dict] = []

    # Score every generated/revised SQL attempt from the agent history.
    history = agent_payload.get("history", [])
    for item in history:
        if item.get("node") not in {"generate_sql", "revise"}:
            continue

        pred_sql = item.get("sql", "")
        pred_ok, pred_rows, pred_error = run_sql(db_id, pred_sql)
        correct = matches(gold_rows, pred_rows) if gold_ok and pred_ok else False

        attempts.append({
            "iteration": len(attempts) + 1,
            "node": item.get("node"),
            "sql": pred_sql,
            "exec_ok": pred_ok,
            "exec_error": pred_error,
            "rows": pred_rows,
            "correct": correct,
        })

    # If the agent returned a final SQL but history was missing/incomplete,
    # still score the final served SQL.
    if not attempts and agent_payload.get("sql"):
        pred_sql = agent_payload["sql"]
        pred_ok, pred_rows, pred_error = run_sql(db_id, pred_sql)
        correct = matches(gold_rows, pred_rows) if gold_ok and pred_ok else False
        attempts.append({
            "iteration": 1,
            "node": "final",
            "sql": pred_sql,
            "exec_ok": pred_ok,
            "exec_error": pred_error,
            "rows": pred_rows,
            "correct": correct,
        })

    return {
        "question": q_text,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "gold_exec_ok": gold_ok,
        "gold_exec_error": gold_error,
        "gold_rows": gold_rows,
        "agent_error": agent_error,
        "agent_ok": agent_payload.get("ok"),
        "agent_final_sql": agent_payload.get("sql"),
        "agent_final_rows": agent_payload.get("rows"),
        "agent_iterations": agent_payload.get("iterations"),
        "latency_seconds": latency,
        "attempts": attempts,
        "final_correct": attempts[-1]["correct"] if attempts else False,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    if total == 0:
        return {
            "num_questions": 0,
            "final_accuracy": 0.0,
            "agent_error_count": 0,
            "gold_error_count": 0,
            "accuracy_by_iteration": {},
            "avg_latency_seconds": 0.0,
        }

    max_iter = max((len(r.get("attempts", [])) for r in results), default=0)
    accuracy_by_iteration: dict[str, float] = {}

    for i in range(1, max_iter + 1):
        correct_count = 0
        for r in results:
            attempts = r.get("attempts", [])
            if not attempts:
                continue

            # Carry forward the last available attempt.
            idx = min(i, len(attempts)) - 1
            if attempts[idx].get("correct", False):
                correct_count += 1

        accuracy_by_iteration[str(i)] = correct_count / total

    final_correct_count = sum(1 for r in results if r.get("final_correct", False))
    agent_error_count = sum(1 for r in results if r.get("agent_error"))
    gold_error_count = sum(1 for r in results if not r.get("gold_exec_ok", False))
    avg_latency = sum(float(r.get("latency_seconds", 0.0)) for r in results) / total

    return {
        "num_questions": total,
        "final_accuracy": final_correct_count / total,
        "correct_final": final_correct_count,
        "agent_error_count": agent_error_count,
        "gold_error_count": gold_error_count,
        "max_iterations_observed": max_iter,
        "accuracy_by_iteration": accuracy_by_iteration,
        "avg_latency_seconds": avg_latency,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
