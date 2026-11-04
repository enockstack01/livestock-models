# Livestock web app

Static page (no build step) served by `python web_app/serve.py` on http://127.0.0.1:8080 (CSP set).

Flow: photo -> Vision API `/v1/detect` (YOLO11x) -> boxes on a canvas -> LLM API `/v1/report` streamed over SSE (Qwen) -> follow-up chat `/v1/chat` -> confirm/correct feedback `/v1/feedback`.
If the LLM API is down the report is the advice-pack fallback and is labelled as such. API URLs and the token are set in the page's settings (localStorage / sessionStorage).
`http://127.0.0.1:8080/#demo` shows a recorded real YOLO11x detection with a MOCK LLM report, without any service running.

Tests: `cd web_app && node --test tests/lib.test.mjs`. Verified in a headless browser only for the demo page; live click-through of upload/stream/chat was exercised via API calls, not by hand.
