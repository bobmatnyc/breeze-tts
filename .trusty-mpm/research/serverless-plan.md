# Breeze TTS 2 on RunPod Serverless — research findings

Scope: what an engineer needs to deploy voice-clone inference behind a RunPod
Serverless endpoint. Findings only — no implementation.

## 1. Clone entrypoint

CLI: `infer.py:70-134` calls `breeze_infer.runtime.load_runtime` then
`FastBreezeStreamingRuntime.iter_audio_chunks` (`models/fast_streaming.py:752`).
Clone is selected implicitly: `--ref-audio` + `--ref-text` without
`--instruction` makes `select_template_name` (`breeze_infer/templates.py:135-145`)
pick the `ref_clone_tata` template (`breeze_infer/templates.py:120-124`).
`prepare_inputs` (`breeze_infer/templates.py:285-361`) builds the tensors;
reference audio goes through `encode_prompt_audio`
(`breeze_infer/audio.py:13-22`) — reads with `soundfile`, downmixes to mono,
calls `audio_tokenizer.encode(wav, sr=sample_rate)` (`Qwen3TTSTokenizer` from
`qwen_tts`) to get `int16` codes shaped `(frames, num_codebooks)`.

`iter_audio_chunks` yields `FastStreamingChunk` (`models/fast_streaming.py:58-64`):
`audio: np.ndarray` (float32 mono PCM), `sample_rate: int`, `codec_frames: int`,
`is_final: bool`. `sample_rate` = `model.config.codec_config.sampling_rate`
(`models/fast_streaming.py:208-209`) — 24 kHz per README.md:165 and the
`X-Sample-Rate` header (`breeze_infer/api.py:261`).

HTTP API (`breeze_infer/api.py`): one route, `POST /v1/audio/speech`
(lines 179-265), `multipart/form-data`: `text` (required), `instruction`
(optional — presence routes to Voice Direction, not Clone), `cfg_scale`
(default 1.0), `ref_audio` (file, optional), `ref_text` (required with
`ref_audio`), `seed` (default 42). Returns `StreamingResponse` of raw mono
PCM16LE at 24 kHz. Also `GET /health` (lines 172-176). Single-concurrency: a
`threading.Lock` (lines 55, 188) rejects a second concurrent request with 409.

**Recommendation**: call the same in-process sequence `infer.py` uses —
`load_runtime` → `FastBreezeStreamingRuntime` → `prepare_inputs` →
`iter_audio_chunks` — directly from a RunPod handler. Skip FastAPI/uvicorn
entirely; RunPod's worker process is already the request boundary. Load the
model once at module scope (mirroring `_load_app`, `api.py:127-158`), then
have `handler(job)` only prepare inputs, iterate chunks, encode to WAV, and
return.

## 2. Model loading

`breeze_infer/runtime.py:68-109` `load_runtime(ckpt_dir, device, attn_implementation)`:
`AutoTokenizer.from_pretrained(ckpt_dir, fix_mistral_regex=False)`, then
`BreezeForConditionalGeneration.from_pretrained(ckpt_dir, dtype=torch.bfloat16, attn_implementation=attn_implementation)`,
`.to(device).eval()`. Audio tokenizer requires `ckpt_dir/audio_tokenizer` to
exist (lines 99-105, hard failure otherwise), loaded via
`Qwen3TTSTokenizer.from_pretrained(...)`. Both CLI (`infer.py:73`) and API
(`api.py:131`) hardcode `attn_implementation="eager"` — flash-attn is never
selected at the attention-backend level despite being built into the image
(§4).

**Weights layout**: one directory (the `model` positional arg) — a standard
HF checkpoint plus a required `audio_tokenizer/` subdirectory bundled inside
it. README.md:62: "All required model components are included in the Breeze
TTS 2 checkpoint."

**On the existing pod** (`/Users/masa/Projects/voice-cloning/runpod/README.md:88-95`):
repo at `/workspace/breeze-tts`, checkpoint at `/workspace/breeze-tts-2`
(matches the `../breeze-tts-2` relative path in README.md and
`pod-synth.sh:26,33,41`), **7.2 GB**. Reference audio/transcripts under
`/workspace/voice/`. Model source: HF repo `BreezeBlue/breeze-tts-2`
(README.md:4). No download command is captured in any script under
`/Users/masa/Projects/voice-cloning/runpod/` or in
`.claude/skills/runpod/SKILL.md` — the checkpoint was evidently fetched
interactively, not via a committed script. That project's README notes a
re-download on a fresh pod "took 15 seconds" (suggests a fast in-datacenter
mirror or `hf_transfer`, not the exact command). **Open question**: confirm
the actual download command (likely `huggingface-cli download
BreezeBlue/breeze-tts-2`).

**Load time**: not recorded anywhere in this repo or the RunPod scripts —
must be measured directly. This is the biggest unknown for cold-start
budgeting.

## 3. GPU / VRAM

README.md:35,44: "Eager inference uses approximately 7.7 GiB of GPU
memory... or 14.4 GiB with `--fast-all`; use a 12 GB GPU for eager or a 24 GB
GPU for the fast path." `--fast-all` is opt-in, off by default in both CLI
(`infer.py:40-41`) and API (`api.py:275-277`) — the CUDA-graph fast path is
optional, only reduces latency/improves RTF, at ~2x VRAM cost plus warmup
time (`_load_app`, `api.py:149-153`, printed as `fast warmup: {ms} ms`).

**Recommendation**: eager mode (no `--fast-all`) on a 24 GB tier GPU
(RTX 4090 / L4 / A5000, §5) — comfortably covers the ~7.7 GiB eager
footprint plus load-time overhead, with headroom this document does not
quantify precisely. A 16 GB tier is plausible but tighter. Skip the fast
path on serverless: CUDA-graph warmup adds fixed per-cold-start latency, and
serverless workers scale to zero, so warmup cost recurs far more often than
on a long-lived pod.

## 4. Container

`docker/Dockerfile` (43 lines): base `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel`
(line 1) — the `-devel` variant is needed only because flash-attn is
compiled from source in the same stage (lines 27-32,
`MAX_JOBS=8 FLASH_ATTN_CUDA_ARCHS=... pip install --no-build-isolation --no-deps "flash-attn==2.8.3"`).
Default `ARG FLASH_ATTN_CUDA_ARCHS=90` (Hopper); overridden to `80` for A100
(`docker/build.sh:6`, `docker/README.md:17-19`).

The flash-attn build is **not conditionally skippable today** — no `ARG`
gate exists around it, and `docker/smoke_check.py:12,20-36` unconditionally
imports and version-checks `flash_attn` as a build-time test (`Dockerfile:36-40`).
Since `attn_implementation` is hardcoded to `"eager"` everywhere it's called
(§2), flash-attn is built but unused by the code paths this plan calls.
Skipping the build (and moving off the `-devel` base) is the largest
available image-size/build-time win for a serverless image, but requires
code changes — an `ARG` gate plus a conditional `smoke_check.py` import —
which this read-only research task does not make; flagged as follow-up work.

Image size drivers, descending: the `-devel` CUDA/cuDNN base, the flash-attn
source build (`ninja-build`, `build-essential`, compiled `.so`), then
`torch`/`torchaudio` wheels. The 7.2 GB checkpoint is correctly not baked in
— `docker/run.sh:16` mounts it read-only, and `.dockerignore` excludes
`outputs`, `*.wav`, `*.pt`/`*.pth`/`*.safetensors`.

Entrypoint today: `ENTRYPOINT ["/opt/breeze-infer/docker/entrypoint.sh"]`
(passthrough `exec "$@"`) with `CMD ["python", "-m", "breeze_infer.api", "--help"]`
— built to run the long-lived FastAPI server. For serverless this becomes a
new `rp_handler.py` (does not exist yet) calling
`runpod.serverless.start({"handler": handler})`, model loaded at import time
(mirroring `_load_app`), with `CMD` changed to
`["python", "-u", "rp_handler.py"]`.

`requirements.txt` pins: `torch==2.9.1`, `torchaudio==2.9.1`,
`qwen-tts==0.1.1`, `transformers==4.57.3`, `numpy>=2.0`, `soundfile>=0.13`,
`fastapi>=0.115`, `uvicorn>=0.30`, `python-multipart>=0.0.18`,
`pytest>=8.0`, `ruff>=0.12`. The `runpod` SDK is **not present** — must be
added. No conflict apparent with the pinned stack; `fastapi`/`uvicorn`/
`python-multipart` become unnecessary for a pure-handler image and can be
dropped, though they cost little relative to the CUDA base.

## 5. RunPod Serverless specifics

**Handler contract** (docs.runpod.io/serverless/workers/handlers/overview):
`runpod.serverless.start({"handler": handler})` registers a function; each
job carries `id` and `input` (client JSON). `/run` (async) accepts input up
to 10 MB; `/runsync` (sync) up to 20 MB. Oversized outputs should not be
inlined — "stash them in cloud storage and return links instead." Binary
audio: docs don't spell out base64 for responses, but combined with the
size-limit guidance and the S3-upload option
(docs.runpod.io/serverless/endpoints/send-requests, `s3Config` for
S3-compatible storage like MinIO/DO Spaces), the two supported paths are
base64-in-JSON (fine under the 10/20 MB cap) or an S3 link. A short clone
clip (seconds, a few MB as WAV) fits base64 comfortably — the simpler
default here.

**Network volumes** (docs.runpod.io/serverless/storage/network-volumes):
mount at `/runpod-volume`. Attaching one pins the endpoint's workers to that
volume's datacenter, which can reduce GPU availability; multiple volumes
across datacenters can widen it (max one per datacenter). Attach via
Endpoint → Manage → Edit Endpoint → Advanced → Network Volumes. Pricing:
~$0.07/GB/month (first TB), $0.05/GB/month beyond. Storing the 7.2 GB
checkpoint here avoids re-downloading into the image or on every cold start.

**Cold start / FlashBoot** (docs.runpod.io/serverless/workers/flashboot):
retains worker state after spin-down so a recently-scaled-down worker
revives faster than a fresh boot; on by default, most effective under
regular traffic. Does not eliminate a true first cold start (model load off
volume/image).

**Pricing** (runpod.io/pricing, hourly, billed per-second): 24 GB tier —
RTX 4090 $1.10/hr, L4/A5000/RTX 3090/24 GB MIG $0.69/hr; 48 GB tier —
A6000/A40 $1.22/hr, L40/L40S/6000 Ada/48 GB MIG $1.75/hr. Flex
(scale-to-zero, standard rate) vs Active (always-on, sales-negotiated
discount) distinguished but per-tier Active rates weren't surfaced in this
fetch. Recommended: **L4/A5000/RTX 3090 class at $0.69/hr** (cheapest 24 GB
option); RTX 4090 at $1.10/hr as a fallback if availability is constrained.

**Build/push and endpoint creation**: standard `docker build` + push to
Docker Hub or GHCR (`docker/build.sh` covers the build half only; push isn't
scripted anywhere in this repo). Console flow: Serverless → New Endpoint →
"Import from Docker Registry" → image ref (e.g.
`docker.io/<user>/breeze-tts-serverless:latest`) → endpoint type "Queue" →
GPU pool → network volume under Advanced. Not confirmed in this pass: GHCR
registry-credential configuration, and `runpodctl create endpoints` flags
(its reference page 404'd) — flagged as an open item below.

## 6. Existing account context

`.env.local` defines exactly one variable name: `RUNPOD_API_KEY` (values not
read, per instructions). No Docker Hub, GHCR, or S3-compatible storage
credential present. `.claude/skills/runpod/SKILL.md` covers only GraphQL
**pod** management (list/provision/start/stop/terminate/status/cost) against
`https://api.runpod.io/graphql` — no serverless-endpoint, network-volume, or
registry operations exist in this skill today. No network volume is
referenced anywhere in the skill or in
`/Users/masa/Projects/voice-cloning/runpod/` — the current setup uses a
pod's own persistent disk (`/workspace`), not a RunPod network volume; one
would be a new resource to provision. No container registry credential
exists in tracked files — the user must choose Docker Hub vs GHCR and supply
that credential outside `.env.local`'s current single key.

## 7. Recommended design

**Files to add** (none exist yet):
- `rp_handler.py` (repo root) — loads the model once at module scope via
  `load_runtime` + `FastBreezeStreamingRuntime` (mirroring `api.py:127-158`
  minus FastAPI); `handler(job)` calls `prepare_inputs` → `iter_audio_chunks`
  → WAV/base64 → return; registers via `runpod.serverless.start`.
- `docker/Dockerfile.serverless` (or a variant of the existing file) — adds
  `runpod` to deps, `CMD` → `["python", "-u", "rp_handler.py"]`; optionally
  gates the flash-attn build behind an `ARG` if that follow-up is taken.
- `docker/build_serverless.sh` — builds and pushes to the chosen registry
  (not scripted today).
- A client script (e.g. `scripts/runpod-clone.sh`) replacing
  `pod-synth.sh`'s `clone` mode — POSTs to the endpoint's `/runsync` (or
  `/run` + poll `/status`) instead of SSHing into a pod.
- Endpoint config, recorded in a new `docker/README.serverless.md` or an
  addition to the RunPod skill (GPU tier, volume, image ref).

**Request/response schema** (proposed):
```
input:  { text, ref_audio_b64, ref_text, seed=42, cfg_scale=1.0 }
output: { audio_b64, sample_rate: 24000 }
```

**Reference audio delivery**: base64 in the request body — simplest,
stateless per request. README's example clips (8-20 s) are comfortably
under the 10 MB `/run` cap as base64 WAV. Reserve the network volume for the
7.2 GB checkpoint only, not per-request reference audio.

**Test plan**:
1. Local CPU import test — a plain `python -c "from rp_handler import
   handler"` import check (not GPU-dependent) before deploying; note
   `docker/smoke_check.py` itself requires `flash_attn`/GPU build, so a
   lighter check is needed for the handler alone.
2. One real serverless invocation using `outputs/Recording_1_ref20.wav`
   (base64-encoded) plus its transcript from
   `outputs/reference_transcripts.json`, via `/runsync`, verifying the
   returned `audio_b64` decodes to a valid 24 kHz WAV.

**Cost estimate**: at the recommended $0.69/hr (≈$0.000192/s), cost per
request is cold-start-plus-load time (unmeasured, §2) plus generation time
(eager-mode RTF not stated in this repo). A dollar figure isn't
responsible to state until load time is measured — top open question below.
As a rough anchor: a generous 60 s combined cold-start + generation ≈
$0.012/request; the real driver is how often a request pays cold-start cost
versus hitting a warm worker (traffic pattern, FlashBoot).

**Open questions for the user**:
1. Exact checkpoint download command/source — not captured anywhere;
   needed to seed a network volume or build-time fetch.
2. Actual cold-load time for `from_pretrained` + audio tokenizer on the
   target GPU — drives cost estimates and the network-volume-vs-baked-in
   decision.
3. Docker Hub vs GHCR, and how the registry credential reaches RunPod — no
   credential exists in this project today.
4. Whether to strip the flash-attn build / drop the `-devel` base (§4) —
   real win, unused by the hardcoded eager path, but a Dockerfile + smoke-check
   code change, deliberately left undone here.
5. `/runsync` vs `/run`+poll — `/runsync` likely suffices for a short
   single-utterance clone, confirm once load time (§2) is known.
6. Confirm `runpodctl create endpoints` flags and the GHCR registry-auth
   console flow directly — not confirmed by this pass (one 404, one partial
   fetch).
