# Livestock LLM API (Qwen wrapper)

FastAPI wrapper around an OpenAI-compatible vLLM server (default model `Qwen/Qwen3.6-27B-FP8`). Code: `src/livestock_llm/`.

It takes a `diagnosis_id` produced by the Vision API, reads the stored detections from the shared database (the services never call each other) and returns a structured, safety-checked report.

| Endpoint | Purpose |
|---|---|
| `POST /v1/report` | Structured report for a diagnosis (`language`, optional `animal.weight_kg`); `stream=true` gives SSE |
| `GET /v1/report/{diagnosis_id}` | Cached report |
| `POST /v1/chat` | Follow-up questions about a diagnosis (SSE optional); user/assistant roles only, 20 messages, 2000 chars each |
| `GET /v1/model`, `/health`, `/ready`, `/metrics` | Ops. `/ready` stays 200 and reports `llm_reachable` |

Behaviour worth knowing:

- **Safety layer is code, not prompt.** Notifiable diseases always get isolate + contact the sector veterinary officer / RAB; doses are stripped unless a weight is given; every medicine carries "use under veterinary supervision"; the disclaimer is always present.
- **Fallback.** If the LLM is down, slow, overloaded or returns invalid JSON twice, the report comes from the bundled advice pack (`source: "advice_pack"`, `degraded: true`). Clients never see an empty result.
- Auth, ownership, rate limit, CORS and error format match the Vision API (shared secret `AGRICODERS_JWT_SECRET`).

## Run

The wrapper reaches Qwen through any **OpenAI-compatible endpoint**, so the model does not have to run on your own GPU. Use a hosted provider that serves Qwen, or your own vLLM server.

```
# hosted provider (use the provider's https URL and the exact Qwen model id from its model list)
set LLM_API_KEY=<your key>          # PowerShell: $env:LLM_API_KEY="<your key>"; environment only, never on the command line
python -m livestock_llm.serve --llm-url https://<provider>/v1 --model <provider's Qwen model id> --port 8002 --dev --data-dir runs/vision_demo
```

Provider differences the wrapper handles:
- **Key**: sent as `Authorization: Bearer`, never logged or put in an error; the server refuses to send it over plain `http://` to a remote host.
- **Unknown fields**: `chat_template_kwargs` is vLLM-specific. If the provider answers HTTP 400, set `LLM_TEMPLATE_KWARGS=0`.
- **Structured output**: tries JSON-schema, steps down to `json_object`, then plain text (the reply is still validated and repaired) when the provider answers 400, and remembers what works. Pin it with `LLM_JSON_MODE`.
- **Rate limits / overload** (429/503): up to 3 retries honouring a short `Retry-After`, then the advice-pack fallback.
- Bad key (401/403) is logged as a configuration error and the API keeps serving fallback reports.

```
# self-hosted vLLM instead
python -m livestock_llm.serve --llm-url http://<vllm-host>:8000/v1 --model Qwen/Qwen3.6-27B-FP8 --port 8002

# local demo with a MOCK model (canned output, clearly labelled, not Qwen)
python -m livestock_llm.dev_mock                       # port 8003
python -m livestock_llm.serve --llm-url http://127.0.0.1:8003/v1 --model mock-qwen --port 8002 --dev --data-dir runs/vision_demo
```

## Verified vs not

Verified: 31 tests (23 API + 8 hosted-provider client: key header, field fallback, 429 retry, key never leaked) (schema, safety rules, fallback, caching, streaming, chat limits, ownership), live runs against the mock server including an outage.
**Not verified: any real Qwen call, hosted or self-hosted.** No provider/key was available here, so the hosted-provider behaviour is tested against `httpx.MockTransport` stand-ins that imitate common OpenAI-compatible responses, not against a real provider. Prompts and JSON-schema output have only been exercised against the mock. Test on a GPU host before relying on report quality. Advice content and translations need veterinary review (Kinyarwanda is not provided yet).
