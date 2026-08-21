"""Score a raw results file; write scored JSON + print a summary table.

Usage: python -m bench.score [--raw results/raw-....json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def args_match(expected: dict, actual: dict) -> bool:
    for key, want in expected.items():
        if key not in actual:
            return False
        got = actual[key]
        if want == "*":
            if got in (None, ""):
                return False
        elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if abs(float(want) - float(got)) > 1e-6:
                return False
        elif got != want:
            return False
    return True


def score_tool_case(case: dict, rec: dict) -> dict:
    expected = case["expected"]["calls"]
    actual = list(rec.get("calls") or [])
    matched = []
    remaining = actual[:]
    for exp in expected:
        hit = next(
            (a for a in remaining if a["tool"] == exp["tool"] and args_match(exp["args"], a["args"])),
            None,
        )
        if hit is None:
            matched.append(False)
        else:
            matched.append(True)
            remaining.remove(hit)
    ok = all(matched) and not remaining  # every expected call found, no extra calls
    return {"pass": ok, "expected_n": len(expected), "actual_n": len(actual)}


def score_json_case(case: dict, rec: dict) -> dict:
    text = rec.get("text") or ""
    strict, lenient_obj = False, None
    try:
        lenient_obj = json.loads(text)
        strict = True
    except json.JSONDecodeError:
        m = FENCE_RE.search(text)
        if m:
            try:
                lenient_obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    schema_valid = False
    checks_pass = True
    if lenient_obj is not None:
        try:
            jsonschema.validate(lenient_obj, case["schema"])
            schema_valid = True
        except jsonschema.ValidationError:
            pass
        for key, want in (case.get("checks") or {}).items():
            if lenient_obj.get(key) != want:
                checks_pass = False
    else:
        checks_pass = False
    return {
        "pass": strict and schema_valid and checks_pass,
        "strict_parse": strict,
        "parsed": lenient_obj is not None,
        "schema_valid": schema_valid,
        "checks_pass": checks_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=None)
    args = parser.parse_args()
    raw_path = args.raw or sorted((ROOT / "results").glob("raw-*.json"))[-1]
    data = json.loads(raw_path.read_text())

    cases = {}
    for kind, fname in [("tool_calling", "tool_calling.jsonl"), ("json_adherence", "json_adherence.jsonl")]:
        for line in (ROOT / "bench/tasks" / fname).read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                cases[(kind, c["id"])] = c

    scored = []
    for rec in data["results"]:
        case = cases[(rec["kind"], rec["case_id"])]
        if "error" in rec:
            s = {"pass": False, "error": rec["error"]}
        elif rec["kind"] == "tool_calling":
            s = score_tool_case(case, rec)
        else:
            s = score_json_case(case, rec)
        scored.append({**rec, "score": s, "tags": case.get("tags", [])})

    summary = {}
    for m in data["models"]:
        rows = [r for r in scored if r["model"] == m["model"]]
        by_kind = {}
        for kind in ("tool_calling", "json_adherence"):
            krows = [r for r in rows if r["kind"] == kind]
            ok = sum(1 for r in krows if r["score"]["pass"])
            lat = [r["latency_ms"] for r in krows if "latency_ms" in r]
            by_kind[kind] = {
                "pass": ok,
                "total": len(krows),
                "errors": sum(1 for r in krows if "error" in r["score"]),
                "median_latency_ms": round(statistics.median(lat)) if lat else None,
            }
            if kind == "json_adherence":
                by_kind[kind]["content_ok"] = sum(
                    1 for r in krows
                    if r["score"].get("schema_valid") and r["score"].get("checks_pass")
                )
        in_tok = sum(r.get("in_tokens", 0) for r in rows)
        out_tok = sum(r.get("out_tokens", 0) for r in rows)
        cost = None
        if m.get("usd_per_mtok_in") is not None:
            cost = round(in_tok / 1e6 * m["usd_per_mtok_in"] + out_tok / 1e6 * m["usd_per_mtok_out"], 4)
        summary[m["model"]] = {
            "label": m["label"],
            **by_kind,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "run_cost_usd": cost,
            "errors": sum(1 for r in rows if "error" in r.get("score", {})),
        }

    out = ROOT / "results/scored-latest.json"
    out.write_text(json.dumps({"raw": raw_path.name, "summary": summary, "cases": scored}, indent=1))

    print(f"{'model':<26} {'tools':>7} {'json strict':>12} {'json content':>13} {'med ms':>8} {'cost $':>8}")
    for model, s in summary.items():
        tc, js = s["tool_calling"], s["json_adherence"]
        cost = f"{s['run_cost_usd']:.3f}" if s["run_cost_usd"] is not None else "n/a"
        print(f"{s['label']:<26} {tc['pass']:>3}/{tc['total']:<3} {js['pass']:>8}/{js['total']:<3} "
              f"{js['content_ok']:>9}/{js['total']:<3} {(tc['median_latency_ms'] or 0):>8} {cost:>8}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
