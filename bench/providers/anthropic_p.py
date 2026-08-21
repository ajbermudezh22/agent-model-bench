"""Anthropic adapter — official SDK, normalized result shape."""

from __future__ import annotations

import time

import anthropic

_client = anthropic.Anthropic()

JSON_SYSTEM = (
    "Respond with ONLY a single JSON object that validates against the JSON Schema "
    "the user provides. No markdown fences, no commentary, no keys beyond the schema."
)


def tool_call(model: str, prompt: str, tools: list[dict]) -> dict:
    t0 = time.monotonic()
    resp = _client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ],
    )
    latency = (time.monotonic() - t0) * 1000
    calls = [
        {"tool": b.name, "args": b.input}
        for b in resp.content
        if b.type == "tool_use"
    ]
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {
        "calls": calls,
        "text": text,
        "latency_ms": latency,
        "in_tokens": resp.usage.input_tokens,
        "out_tokens": resp.usage.output_tokens,
    }


def json_task(model: str, prompt: str, schema: dict) -> dict:
    import json as _json

    t0 = time.monotonic()
    resp = _client.messages.create(
        model=model,
        max_tokens=1024,
        system=JSON_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"JSON Schema:\n{_json.dumps(schema)}\n\nTask: {prompt}",
            }
        ],
    )
    latency = (time.monotonic() - t0) * 1000
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {
        "text": text,
        "latency_ms": latency,
        "in_tokens": resp.usage.input_tokens,
        "out_tokens": resp.usage.output_tokens,
    }
