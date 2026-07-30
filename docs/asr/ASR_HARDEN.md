# ASR / voice hardening (optional)

**Status:** backlog / decide later  
**Not required** for normal solo use with headphones and barge off.

Use this doc as a todo checklist. Pick items only when real pain shows up.

---

## Purpose

This hardening plan is industrial-style polish the audit called out as gaps vs pro stacks:

| Gap | Action items |
|-----|---------|
| Little production telemetry | Metrics |
| Weak local AEC (speakers → mic) | Local AEC with TTS reference |
| Raw PCM on WebSocket | Opus (or similar) if bandwidth hurts |
| Utterance-final only (no partials) | Streaming ASR for live partials |

None of these require abandoning SenseVoice for the final transcript.

---

## Scope

### In scope

1. **Metrics** — counters / timings for empty rate, RTF, barge rate, speaker scores  
2. **Local AEC** — on-device echo cancel for robot/`parec` (and optionally stronger Web path)  
3. **Opus (or codec) on WS** — compress browser → Jetson uplink  
4. **Streaming ASR partials** — incremental hypotheses while user still talks  

### Out of scope (unless revisited later)

- SenseVoice **finetune** (use S2 corrections / aliases first)  
- Replacing `parec` with WebRTC for local mic  
- Dropping server Silero or dual-VAD policy (see `VAD_POLICY.md`)  
- Contact-center grade confidence / hotword servers  
- Mixing into memory M-* PRs  

---

## Decision guide — do I need this?

| Your situation | Suggestions |
|----------------|--------------|
| Solo, headphones, `BARGE_IN_ENABLED=0` | **None** |
| Want numbers to tune yaml | **Metrics only** |
| Speakers + barge on + still self-cuts after S3 | **Local AEC** (after metrics) |
| Slow uplink / remote WebUI | **Opus** |
| “Feels laggy” waiting for silence + full decode | **Streaming partials** (largest project) |

**Default recommendation:** leave it closed until one of the middle rows is true.

---

## Items (priority order)

### 1. Metrics (low effort, do first if anything)

**Why:** You cannot tune threshold / echo / speaker gate without rates and timings.

**What to log (examples):**

| Metric | Meaning |
|--------|---------|
| Empty / reject rate | Utterances → `""` or gate drop |
| Mean utterance ms | Speech length distribution |
| ASR RTF | Decode time / audio duration |
| Speech-end → text latency | Hangover + SenseVoice |
| Barge-in rate | Fires per TTS session |
| Speaker score histogram | When verify enabled |

**Where:** lightweight counters in `listen.py` / barge path; optional periodic log or small JSONL under user state. No dashboard required for v1.

**Done when:** you can answer “how often empty?” and “how slow is final text?” from logs after a day of use.

---

### 2. Local AEC (medium effort)

**Why:** Browser already has `echoCancellation: true`; speakers still leak into the mic. S3 echo **guard** only blocks barge for ~450 ms — it does not clean ASR audio.

**Paths:**

| Path | Approach |
|------|----------|
| WebUI | Keep browser AEC; optional stronger product logic already in S3 |
| Local robot | Need **reference** = TTS playout (loopback) + algo |

**Options for local:**

- PulseAudio `module-echo-cancel` (system-level)  
- WebRTC APM / similar with TTS reference mixed into the cancel path  
- SpeexDSP AEC (classic, more DIY)  

**Hard parts:** stable reference delay, nonlinear speaker distortion, Jetson CPU budget.

**Done when:** with speakers + barge on, self-voice no longer drives barge or garbles the user’s transcript in normal room tests.

**Skip if:** you mostly use wireless headphones.

---

### 3. Opus on WebSocket (low–medium effort)

**Why:** 16 kHz float32 PCM is simple but fat on weak uplinks.

**What:** encode browser frames → Opus (or similar); decode on Jetson before VAD/ASR.

**When:** remote WebUI over constrained links; not needed for localhost / LAN.

**Done when:** uplink bandwidth drops without measurable ASR quality loss.

---

### 4. Streaming ASR partials (high effort)

**Why:** Industrial UIs show partial text while the user talks. Aiko waits for Silero end → full SenseVoice decode.

**Pattern (hybrid edge):**

```text
Mic → Silero still endpoints
     → streaming model (Zipformer / Paraformer streaming, sherpa) → partials to UI
     → SenseVoice (or streaming final) for committed text
```

**Keep:** SenseVoice for multilingual final / quality if streaming model is weaker on zh/ja/ko/yue.

**Hard parts:** model load RAM on Jetson, partial vs final consistency, barge interaction, UI token stream wiring.

**Done when:** UI shows stable partials during speech and a clean final after endpoint — without tanking RTF on device.

**Skip if:** utterance-final latency is acceptable.

---

## Explicit non-goals (for now)

- Finetune SenseVoice for “Aiko” / “OppaAI” (S2 is enough until proven otherwise)  
- Double Silero in the browser  
- WebRTC as the only transport for local mic  

---

## Todo checklist (copy into your list)

```text
[ ] Decide: any pain after a week of real use?
[ ] a) Metrics — empty rate, RTF, barge rate, speaker scores
[ ] b) Local AEC — only if speakers + barge still bad
[ ] c) Opus on WS — only if uplink bandwidth hurts
[ ] d) Streaming partials — only if felt latency is the main complaint
[ ] Revisit SenseVoice finetune only if S1+S2 still fail on names
```

---

## References

- Dual VAD policy: `sensory/DUAL_VAD_POLICY.md`  
- Debug checklist: `docs/DEBUG_AUDIO.md`  
- Config: `config/sensory.yaml`  
- Capture / barge / ASR: `sensory/listen.py`, `interface/webui/static/vad.js`  
