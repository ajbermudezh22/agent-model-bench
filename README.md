# agent-model-bench

**Which model should power your agent?** A small, honest benchmark of the two
capabilities agent frameworks actually depend on — tool calling and strict JSON
output — plus the latency and cost that decide what you can afford in an inner loop.

**[→ Live report](https://ajbermudezh22.github.io/agent-model-bench/)**

## What it measures

**Tool calling (20 cases).** Given 5 tool schemas and a user request: does the model
pick the right tool with the right arguments? Cases include format coercion (city →
IATA code, "nonstop" → `max_stops: 0`), knowledge resolution ("capital of Portugal" →
Lisbon), parallel calls, German input, and — the interesting part — *traps*:
underspecified requests and no-matching-tool requests where the correct behavior is
to NOT call anything instead of hallucinating arguments. Exact tool + argument match,
over-calling penalized.

**JSON adherence (12 cases).** Extraction/construction against a JSON Schema with the
instruction "raw JSON only, no fences." Scored twice:
- **strict** — the response parses as-is (did it follow the instruction?)
- **content** — schema-valid after stripping markdown fences (is the JSON right?)

The gap between the two is formatting discipline, and it turns out to be the most
model-differentiating metric in the suite: some models produce perfect JSON content
but wrap it in ```` ```json ```` fences no matter what you tell them — which is
exactly the thing that breaks naive `json.loads` glue code in production.

## Run it

```sh
export ANTHROPIC_API_KEY=... GEMINI_API_KEY=...
uv sync
uv run python -m bench.run      # ~$0.4 in API calls, provider-throttled
uv run python -m bench.score
uv run python -m bench.report   # -> docs/index.html
```

Providers are pluggable: a provider module implements `tool_call()` and `json_task()`
returning a normalized record (see `bench/providers/`). Models and list prices live in
`bench/models.json`.

## Honest limitations

- 32 cases is a screen, not a leaderboard — differences of 1-2 cases are noise.
- Single-turn only; no multi-turn tool conversations.
- No system-prompt tuning per model; every model gets the same instructions.
- Gemini free-tier pricing/quota made costs non-comparable there (marked n/a).
