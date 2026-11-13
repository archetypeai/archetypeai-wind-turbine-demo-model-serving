# Migration: Omega 1.4 on the Direct Query API → Omega 1.5 on the Inference API

This project is the [Penmanshiel wind turbine demo](https://github.com/archetypeai/archetypeai-wind-turbine-demo) with its Omega embedding calls moved from Newton Omega 1.4 on the Direct Query API to Newton Omega 1.5 on the Inference API. Everything else is unchanged: replay, reference windows, scaler, KNN, alert rules and UI.

## Summary

- **Endpoint and model:** the app opens one session on model `omega-embedding` (Omega 1.5) with `POST /api/inference/sessions`, then queries it with `POST /api/inference/sessions/{id}/query`: one request per window, carrying all channels.
- **Speed:** about 4.6× faster per window and 3–6× faster for the full classification run.
- **Window size:** without a session, the model only takes 1024-sample windows. A session sets `window_size` and `step_size`, so the demo embeds its real 128-row windows.
- **New library:** Omega 1.5 embeddings aren't compatible with Omega 1.4 ones, so `library.json` was rebuilt.
- **Outcome on the demo:** Omega 1.5 raises **no alerts on the healthy turbine** (Omega 1.4 raised one), but it flags WT01's fault **about 1.8 days later** (one window).

## Direct Query API vs Inference API

| | Direct Query API | Inference API |
|---|---|---|
| Endpoint | `POST /api/lens/query` | `POST /api/inference/query`, or `POST /api/inference/sessions/{id}/query` in a session |
| Model | `OmegaEncoder::omega_embeddings_1_4` | `omega-embedding` |
| Omega version | 1.4 | 1.5 |
| Window length | 16–1024 samples; shorter is padded **and masked** (with a warning); longer keeps the last 1024 | 1024 without a session. **Any length from 16 to 1024 in a session** (`values.window_size` / `step_size`); shorter than 1024 is padded and masked inside the model. Input shorter than the window returns `{"response": [], "success": false}` with no message; longer input is cut into back-to-back windows, one row each, and the rest is dropped |
| `normalize_input` | Supported (`true` / `false`) | Accepted, no effect. The model z-scores each window itself and passes the window's original scale to the encoder |
| Channels per request | One per request was the safe pattern, so 4 calls per window | All channels in one request (4 × 128 samples is a few KB; the limit is 2 MiB) |
| `response.response` | Flat `[768]` for one channel | `[1, channels × 768]`: one row per window, channels concatenated in order |
| Latency, 4 ch × 1024 (median) | **about 2.0 s** (4 parallel calls) | **about 0.4 s** (1 call) |

### Embeddings are not interchangeable

For the same 4-channel window, the Omega 1.4 vectors (per channel, concatenated) and the Omega 1.5 vector have a **cosine similarity close to 0** (between −0.03 and 0.02 in our tests): essentially unrelated. A KNN library, threshold or downstream model built on one version can't score embeddings from the other. Rebuild with `python build_library.py`. The `library.json` fingerprint records the model and session values, so a stale library is detected and rebuilt automatically.

## What changed in the code

| File | Change |
|---|---|
| `newton_client.py` | `OMEGA_MODEL = "omega-embedding"`, `SESSIONS_PATH = "/inference/sessions"`. The per-channel `_embed_channel` + thread-pool fan-out is replaced by a single `_embed_window` call carrying all 4 channels. `_get_session()` opens one session with `window_size` / `step_size` = 128, reopens it on `session_not_running`, and an `atexit` hook cancels it. `success: false` and wrong-sized vectors are treated as failures and retried. The library fingerprint includes the session values. `OMEGA_MAX_CONCURRENCY` removed. |
| `library.json` | Rebuilt with Omega 1.5 on 128-sample session windows: 18 healthy + 8 fault windows, as before. |
| `app.py`, `build_library.py` | `load_dotenv(override=True)`: the project `.env` wins over `ATAI_API_*` variables exported in your shell. |
| `templates/index.html` | Subtitle names Omega 1.5 + local KNN. |
| `README.md` | Architecture diagram and text updated, plus a "Window size and sessions" section. |

### Window size: sessions

The demo classifies 128-row windows (about 21 h at 10-minute cadence). 1024 real rows (about 7 days) wouldn't work, because the only fault reference is WT05's ~36 h outage (~250 rows), and zero-padding by hand would be read as signal.

- **A session sets the window.** `POST /api/inference/sessions` with `"values": {"window_size": 128, "step_size": 128}`, and each 128-row slice is sent as is. An unknown value is refused with `422 invalid_session_values`.
- **Short windows are handled properly.** Omega 1.5 z-scores over the real samples, left-pads to 1024 and masks the padding out of attention and pooling.
- **Session limits:** a session ends when it expires or the model is redeployed, and then answers `409 session_not_running`; the app opens a new one. An organization can hold 16 open sessions on the model (the 17th returns `429 session_limit_reached`), so the app cancels its session on exit.

An earlier version of this migration stretched each 128-row window to 1024 points instead of using a session. It produced the same alerts as the session version below.

## Results on the demo

Both pipelines were run over the same 102 windows (51 per turbine, Sep 1 → Dec 1 2019), with the app's own alert rule (a verdict must lead by ≥ 3 of 5 votes). The Omega 1.5 results were the same on every deployment we ran them on.

| | Omega 1.4, Direct Query API | Omega 1.5, Inference API (128-sample session) |
|---|---|---|
| WT01 fault detected | Window starting **Nov 4 00:00** (5–0) | Window starting **Nov 5 18:40** (5–0) |
| WT01 recovery | Nov 7 13:20 | Nov 7 13:20 |
| WT09 (healthy peer) alerts | **1**: Oct 17 05:20 → recovered Oct 19 | **0** |
| Errors | 0 | 0 |
| Wall time, full run | 258 s | **about 45–75 s** |

Context from the raw SCADA data:

- **WT01** was already stopped from **Nov 3** (pitch about 91°, power about −2 kW, wind 5–14 m/s). So the Nov 4 window is a real fault, and Omega 1.5 calls it healthy with a 5–0 vote: a genuine ~1.8-day detection delay.
- **WT09 Oct 17–18:** blades feathered (pitch 80–92°), near-zero power, in light wind (3–7 m/s). This looks like an idle or low-wind stop rather than a fault, so it's ambiguous whether Omega 1.4's alert there is wrong.

**Model or window handling?** Omega 1.4 run on the same stretched 1024-point windows, with a library rebuilt the same way, still catches WT01 on Nov 4 (5–0). Omega 1.5 on 128-sample session windows gives the same result as the stretched version. So the later detection comes from Omega 1.5 itself, not from how windows are sent.

## Reproducing

```bash
.venv/bin/python build_library.py   # rebuild library.json with Omega 1.5
.venv/bin/python app.py             # http://127.0.0.1:5050 → Start
```
