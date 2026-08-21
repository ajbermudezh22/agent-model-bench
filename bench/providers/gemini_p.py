"""Gemini adapter — raw REST (generateContent), normalized result shape."""

from __future__ import annotations

import copy
import json
import os
import time

import httpx

BASE = "https://generativelanguage.googleapis.com/v1beta/models"

JSON_SYSTEM = (
    "Respond with ONLY a single JSON object that validates against the JSON Schema "
    "the user provides. No markdown fences, no commentary, no keys beyond the schema."
)


def _post(model: str, body: dict) -> tuple[dict, float]:
    key = os.environ["GEMINI_API_KEY"]
    t0 = time.monotonic()
    resp = httpx.post(f"{BASE}/{model}:generateContent?key={key}", json=body, timeout=120)
    latency = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    return resp.json(), latency


def _clean_schema(schema: dict) -> dict:
    """Gemini's functionDeclarations reject some JSON Schema keywords."""
    schema = copy.deepcopy(schema)

    def strip(node):
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            for v in node.values():
                strip(v)
        elif isinstance(node, list):
            for v in node:
                strip(v)

    strip(schema)
    return schema


def tool_call(model: str, prompt: str, tools: list[dict]) -> dict:
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": _clean_schema(t["input_schema"]),
                    }
                    for t in tools
                ]
            }
        ],
    }
    data, latency = _post(model, body)
    parts = data["candidates"][0].get("content", {}).get("parts", [])
    calls = [
        {"tool": p["functionCall"]["name"], "args": p["functionCall"].get("args", {})}
        for p in parts
        if "functionCall" in p
    ]
    text = "".join(p.get("text", "") for p in parts)
    usage = data.get("usageMetadata", {})
    return {
        "calls": calls,
        "text": text,
        "latency_ms": latency,
        "in_tokens": usage.get("promptTokenCount", 0),
        "out_tokens": usage.get("candidatesTokenCount", 0),
    }


def json_task(model: str, prompt: str, schema: dict) -> dict:
    body = {
        "systemInstruction": {"parts": [{"text": JSON_SYSTEM}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"JSON Schema:\n{json.dumps(schema)}\n\nTask: {prompt}"}],
            }
        ],
    }
    data, latency = _post(model, body)
    parts = data["candidates"][0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    usage = data.get("usageMetadata", {})
    return {
        "text": text,
        "latency_ms": latency,
        "in_tokens": usage.get("promptTokenCount", 0),
        "out_tokens": usage.get("candidatesTokenCount", 0),
    }
