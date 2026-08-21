"""Run every case against every model; write one raw results JSON.

Usage: python -m bench.run [--only-model SUBSTR] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.providers import anthropic_p, gemini_p

ROOT = Path(__file__).resolve().parent.parent
PROVIDERS = {"anthropic": anthropic_p, "gemini": gemini_p}
RETRIES = 5
GEMINI_GAP_S = 7.0  # free-tier RPM protection
_gemini_lock = None  # set in main


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_one(model_cfg: dict, kind: str, case: dict, tools: list[dict]) -> dict:
    provider = PROVIDERS[model_cfg["provider"]]
    rec = {"model": model_cfg["model"], "kind": kind, "case_id": case["id"]}
    for attempt in range(RETRIES):
        try:
            if model_cfg["provider"] == "gemini" and _gemini_lock is not None:
                with _gemini_lock:
                    time.sleep(GEMINI_GAP_S)
                    if kind == "tool_calling":
                        out = provider.tool_call(model_cfg["model"], case["prompt"], tools)
                    else:
                        out = provider.json_task(model_cfg["model"], case["prompt"], case["schema"])
            elif kind == "tool_calling":
                out = provider.tool_call(model_cfg["model"], case["prompt"], tools)
            else:
                out = provider.json_task(model_cfg["model"], case["prompt"], case["schema"])
            rec.update(out)
            return rec
        except Exception as exc:
            is_429 = "429" in str(exc)
            if attempt == RETRIES - 1:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                return rec
            time.sleep(30 * (attempt + 1) if is_429 else 2**attempt * 2)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-model", default="")
    parser.add_argument("--limit", type=int, default=0, help="First N cases per suite")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    models = json.loads((ROOT / "bench/models.json").read_text())
    if args.only_model:
        models = [m for m in models if args.only_model in m["model"]]
    tools = json.loads((ROOT / "bench/tasks/tools.json").read_text())["tools"]
    suites = {
        "tool_calling": load_jsonl(ROOT / "bench/tasks/tool_calling.jsonl"),
        "json_adherence": load_jsonl(ROOT / "bench/tasks/json_adherence.jsonl"),
    }
    if args.limit:
        suites = {k: v[: args.limit] for k, v in suites.items()}

    jobs = [
        (m, kind, case)
        for m in models
        for kind, cases in suites.items()
        for case in cases
    ]
    print(f"{len(jobs)} calls across {len(models)} models")

    import threading

    global _gemini_lock
    _gemini_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda j: run_one(j[0], j[1], j[2], tools), jobs))

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or ROOT / f"results/raw-{stamp}.json"
    out.write_text(json.dumps({"generated": stamp, "models": models, "results": results}, indent=1))
    errors = sum(1 for r in results if "error" in r)
    print(f"wrote {len(results)} records ({errors} errors) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
