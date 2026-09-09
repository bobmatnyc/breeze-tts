# RunPod Serverless deployment

Voice cloning behind a scale-to-zero HTTP endpoint. The worker is
[`rp_handler.py`](../rp_handler.py); the client is
[`scripts/runpod_clone.py`](../scripts/runpod_clone.py).

## Why a serverless worker instead of the FastAPI server

`breeze_infer/api.py` wraps the runtime in uvicorn because a pod runs for hours.
A RunPod worker process is already the request boundary, so `rp_handler.py` runs
the same sequence — `load_runtime` → `prepare_inputs` → `iter_audio_chunks` —
with no HTTP layer of its own. The model loads once at import, and each job only
prepares inputs, streams chunks, and encodes a WAV.

## Building the image

One Dockerfile serves both targets. A second file would duplicate 40 lines of
apt and pip setup for a two-line delta and drift the moment either changed, so
the differences are build args instead:

| Arg | Pod / dev default | Serverless build | Effect |
| --- | --- | --- | --- |
| `BASE_IMAGE` | `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` | `…-cudnn9-runtime` | 9.6 GB base drops to 4.5 GB |
| `BUILD_FLASH_ATTN` | `1` | `0` | Skips the flash-attn source build |
| `INSTALL_SERVERLESS_DEPS` | `0` | `1` | Adds `runpod` and `huggingface-hub` |

Flash-attention is off for serverless because nothing selects it: both
`infer.py:73` and `breeze_infer/api.py:131` hardcode
`attn_implementation="eager"`, so the compiled kernels are never reached. That
also removes the only reason for the `-devel` base. `docker/smoke_check.py`
treats `flash_attn` as optional, so the gated build still runs its own gate.

`docker/build.sh` and `docker/run.sh` are unchanged, and with the default args
the pod/dev image installs exactly what it did before — same base, same
flash-attn build, no serverless dependencies. One thing does differ: the default
`CMD` is now `python -u rp_handler.py` instead of `python -m breeze_infer.api
--help`. `CMD` cannot be varied by a build arg without an entrypoint shim, and
the pod path never reaches it — `docker/run.sh:18` passes an explicit
`python -m breeze_infer.api …` command that overrides `CMD` on every run. So the
change is to a default the pod path does not use, not to how it behaves.

Build locally:

```bash
docker build --file docker/Dockerfile \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime \
  --build-arg BUILD_FLASH_ATTN=0 \
  --build-arg INSTALL_SERVERLESS_DEPS=1 \
  --platform linux/amd64 \
  --tag ghcr.io/<owner>/breeze-tts-serverless:latest .
```

In practice CI does this.
[`.github/workflows/serverless-image.yml`](../.github/workflows/serverless-image.yml)
builds `linux/amd64` and pushes to GHCR on `workflow_dispatch` and on pushes to
`feat/runpod-serverless`, tagging both the git SHA and `latest`. It uses
`GITHUB_TOKEN` with `packages: write`, so no registry secret is needed. A
"Free runner disk space" step drops the preinstalled .NET/Android/GHC trees
first — the CUDA base plus the torch wheels overflow what an `ubuntu-latest`
runner leaves free.

The image name is `ghcr.io/${{ github.repository_owner }}/breeze-tts-serverless`,
so the workflow publishes under whichever account runs it.

**The GHCR package must be public** for RunPod to pull it without credentials.
The `org.opencontainers.image.source` label links the package to this
repository so it inherits the repository's access; from a public repository
that makes the package publicly pullable. Verify anonymously before creating
the endpoint:

```bash
tok=$(curl -s "https://ghcr.io/token?scope=repository:<owner>/breeze-tts-serverless:pull&service=ghcr.io" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $tok" \
  -H 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json' \
  https://ghcr.io/v2/<owner>/breeze-tts-serverless/manifests/latest
```

Include `application/vnd.oci.image.manifest.v1+json` in `Accept`. The build is
single-platform with `provenance: false`, so the tag resolves to a bare OCI
manifest rather than an index; omitting that type returns 404 on a package that
is in fact public.

There is no REST endpoint for package visibility — if the package does come out
private, it can only be flipped in the GitHub UI (package → Package settings →
Change visibility), or worked around with a RunPod container registry
credential (`POST /v1/containerregistryauth`) passed as the template's
`containerRegistryAuthId`.

## The deployed endpoint

Created against `https://rest.runpod.io/v1` with the account's `RUNPOD_API_KEY`.

| Setting | Value |
| --- | --- |
| Endpoint id | `53bev6svysh8g4` |
| Endpoint name | `breeze-tts-clone` |
| Template id | `lbawlul4gz` |
| Image | `ghcr.io/bobmatnyc/breeze-tts-serverless:latest` |
| Image digest at live test | `sha256:7afcf1bda51e5e82721a8c9b1f2f02f39b366f6c991190c3ae7640f9d9e5825b` |
| Immutable tag for that build | `…-serverless:66c4b2221e276aa2c6a5489a9390b0ffccc18500` |
| Network volume | `breeze-tts-2-weights`, id `wcc9h8jf77`, 10 GB, `EU-RO-1` |
| GPU pool | `NVIDIA GeForce RTX 4090`, then `NVIDIA RTX A5000`, `NVIDIA L4` |
| Compute type | GPU, 1 per worker |
| Data centers | `EU-RO-1`, pinned by the network volume |
| FlashBoot | on |
| Workers | min 0, max 1 |
| Idle timeout | 30 s |
| Execution timeout | 900000 ms |
| Container disk | 30 GB |

The template points at `:latest`, which every build moves, so a worker started
after a later build runs that build instead. Point the template at the commit
tag to pin a specific image; the digest above is the one the live test ran.

Min workers is 0, so the endpoint bills nothing while idle. The network volume
is billed continuously at roughly $0.07/GB/month — about **$0.70/month** for
10 GB — whether or not a worker is running.

Eager inference needs about 7.7 GiB (README.md:44), so a 24 GB tier is
comfortable. `--fast-all` is deliberately not used: its CUDA-graph warmup is a
fixed cost paid on every cold start, and serverless workers scale to zero.

### Recreating it

```bash
# 1. Network volume, in a datacenter that has the GPU tier you want.
curl -X POST https://rest.runpod.io/v1/networkvolumes \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"breeze-tts-2-weights","dataCenterId":"EU-RO-1","size":10}'

# 2. Serverless template pointing at the image.
curl -X POST https://rest.runpod.io/v1/templates \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"breeze-tts-serverless","imageName":"ghcr.io/<owner>/breeze-tts-serverless:latest",
       "isServerless":true,"containerDiskInGb":30,"volumeMountPath":"/runpod-volume",
       "env":{"PYTHONUNBUFFERED":"1"}}'

# 3. Endpoint. templateId is required; the image lives on the template.
curl -X POST https://rest.runpod.io/v1/endpoints \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"breeze-tts-clone","templateId":"<id>","computeType":"GPU",
       "gpuTypeIds":["NVIDIA GeForce RTX 4090","NVIDIA RTX A5000","NVIDIA L4"],
       "dataCenterIds":["EU-RO-1"],"networkVolumeId":"<volume id>",
       "workersMin":0,"workersMax":1,"idleTimeout":30,
       "executionTimeoutMs":900000,"flashboot":true}'
```

Attaching a network volume pins the endpoint's workers to that volume's
datacenter, which narrows GPU availability. `dataCenterIds` is set to match so
the constraint is explicit rather than implied.

## Model weights

The worker keeps the checkpoint at `/runpod-volume/breeze-tts-2/`, downloading
`BreezeBlue/breeze-tts-2` from Hugging Face on first boot so it seeds itself and
no separate seeding pod is needed. The repository is public and ungated; no
`HF_TOKEN` is required. Set `BREEZE_CHECKPOINT_DIR` to load from somewhere else.

Readiness is the `.breeze-complete` marker written after a successful download,
not the presence of `audio_tokenizer/` — the directory `load_runtime` hard-fails
without (`breeze_infer/runtime.py:99-105`). `snapshot_download` populates the
tree incrementally, so a worker killed mid-download would otherwise leave a
checkpoint that looks finished forever. The download runs into a staging
directory under an `flock` on the volume and is renamed into place only when it
completes, so a concurrent worker waits rather than racing and an interrupted
attempt is discarded and retried on the next start.

Staging beside the target would mean two copies on the volume at once — 14.4 GB
where only 10 GB exists. An incomplete checkpoint is unusable by definition, so
it is deleted before staging starts, along with any staging directory a killed
attempt leaked; peak usage stays at one copy. Sizing the volume for two copies
instead would work, at double the monthly cost for space that is idle except
during a re-download.

## Invoking it

```bash
# A registered voice (see "Named voices" below):
python scripts/runpod_clone.py clone \
  --endpoint-id 53bev6svysh8g4 --voice bob \
  --text "This is a test of voice cloning using my actual voice recordings" \
  --output outputs/clone.wav

# Or an inline reference clip:
python scripts/runpod_clone.py clone \
  --endpoint-id 53bev6svysh8g4 \
  --text "This is a test of voice cloning using my actual voice recordings" \
  --ref-audio outputs/Recording_1_ref20.wav \
  --ref-text "<exact transcript of the reference clip>" \
  --output outputs/clone.wav
```

The client reads `RUNPOD_API_KEY` from the environment, falling back to
`.env.local` at the repository root. It posts to `/runsync` by default; pass
`--async` for long texts, which submits to `/run` and polls `/status`. It prints
the endpoint's own `delayTime` (queue plus cold start) and `executionTime`
alongside the worker's model-load and generation timings.

### Request and response

```jsonc
// input — exactly one of "voice" or the ref_audio_b64/ref_text pair
{ "op": "clone", "text": "...", "voice": "bob",
  "seed": 42, "cfg_scale": 1.0 }

// output
{ "audio_b64": "<base64 WAV>", "sample_rate": 24000,
  "audio_seconds": 4.4, "truncated": false, "decode_steps": 331,
  "prepare_seconds": 0.124, "generation_seconds": 13.046,
  "model_load_seconds": 4.664 }
```

`cfg_scale` must be 1.0. The `ref_clone_tata` template defines no negative
prompt (`breeze_infer/templates.py:120-124`), so `prepare_inputs` rejects any
other guidance scale (`breeze_infer/templates.py:342-346`); the worker says so
up front rather than failing mid-request.

Reference audio travels base64 in the request body, capped at 6 MB decoded —
`/run` accepts 10 MB of JSON and `/runsync` 20 MB, and base64 inflates by 4/3.
A 20 s clip is well under that. The response WAV rides back the same way.

Rejected inputs come back as `{"error": "..."}` with HTTP 200 and RunPod status
`COMPLETED`; the client checks for the key and exits non-zero.

## Cost and measured timings

Billing is per second of worker runtime at the GPU tier's rate: RTX 4090 is
$1.10/hr ($0.000306/s), the A5000/L4 class $0.69/hr ($0.000192/s). RunPod does
not report which member of the pool served a given job, so the figures below are
a range across the pool.

Two live calls on the deployed endpoint, reference clips from `outputs/`:

| | Cold call (40 s reference) | Warm call (20 s reference) |
| --- | --- | --- |
| `delayTime` | 76399 ms | 1369 ms |
| `executionTime` | 35411 ms | 13821 ms |
| Worker model load | 4.66 s | 4.66 s (same worker) |
| Generation | 33.72 s | 13.05 s |
| Audio returned | 11.04 s | 4.40 s |
| Real-time factor | 3.05 | 2.97 |
| Billed worker time | 111.8 s | 15.2 s |
| Cost at $0.69-$1.10/hr | $0.021 - $0.034 | $0.003 - $0.005 |

`delayTime` on the cold call covers the image pull, container start, the
one-time 7.2 GB checkpoint download onto the empty network volume, and the model
load. Only the checkpoint download is one-time per volume; later cold starts skip
it. `model_load_seconds` measures `load_runtime` alone, after the checkpoint is
resolved — 4.66 s off the network volume.

Eager-mode RTF is about 3.0 on this pool: roughly 1 s of GPU time per 3 s of
audio. Warm steady-state cost is therefore about $0.0006-$0.001 per second of
generated speech.

Idle costs nothing beyond the network volume's ~$0.70/month.

FlashBoot is on: it revives a recently scaled-down worker faster than a fresh
boot, but it does not remove the first true cold start. If cold starts dominate
your traffic, raising `workersMin` to 1 trades about $17-26/day of idle billing
for their removal.

## Tearing it down

```bash
curl -X DELETE https://rest.runpod.io/v1/endpoints/<endpoint id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/templates/<template id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
curl -X DELETE https://rest.runpod.io/v1/networkvolumes/<volume id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

Delete in that order — an endpoint holds its template, and the volume cannot be
released while an endpoint references it. The endpoint must be scaled to zero
workers before deletion; setting `workersMin: 0` and letting the idle timeout
elapse is enough. Deleting the volume destroys the cached checkpoint, so the
next deployment re-downloads it.

Leaving the endpoint in place costs nothing while idle. The volume keeps
billing, so delete it too if the deployment is finished for good.

## Named voices

Sending a multi-megabyte reference clip on every request is wasteful when the
same speaker repeats, and an OpenAI-style client has nowhere to put one — it
sends a voice *name*. Registering a voice once turns every later request into a
short JSON body.

Voices live on the network volume under `/runpod-volume/voices/<name>/`, each
holding `reference.wav`, `transcript.txt` and a `meta.json` recording the
creation time, source filename, duration, sample rate and the SHA-256 of the
WAV. Names must match `[a-z0-9_-]{1,32}` — they become path segments and travel
in request bodies, so the character set is narrower than merely path-safe.

Writes take an `flock` on the volume and build into a staging directory that is
renamed into place, with a `.breeze-complete` marker written last. A worker
killed mid-write therefore leaves a directory with no marker, which reads as
absent rather than as a usable voice. The checkpoint download uses the same
discipline, so a second worker cold-starting on a fresh volume waits for the
first rather than racing it.

```bash
python scripts/runpod_clone.py register --endpoint-id <id> \
  --name bob \
  --ref-audio outputs/Recording_1_ref40.wav \
  --ref-text "<exact transcript>" [--overwrite]

python scripts/runpod_clone.py list   --endpoint-id <id>
python scripts/runpod_clone.py delete --endpoint-id <id> --name bob
python scripts/runpod_clone.py clone  --endpoint-id <id> \
  --voice bob --text "Hello" --output outputs/hello.wav
```

The job `input` carries an `op`: `clone` (the default), `register_voice`,
`list_voices`, `delete_voice`. A clone takes **exactly one** voice source —
either `voice` or the `ref_audio_b64`/`ref_text` pair; supplying both or neither
is an error. An unknown voice name fails closed with a message listing what is
registered.

## Truncation

`iter_audio_chunks` stops on end-of-speech, on `max_new_tokens`, or on
`max_seq_len` (`models/fast_streaming.py:850-854,878`). Every path marks the
last chunk `is_final`, so that flag cannot tell a finished utterance from a
severed one. The worker counts decode steps through the token observer instead
and returns `truncated` alongside `decode_steps`; the client prints a warning
when it is true, and the shim sets an `X-Truncated` response header.

## Reading long texts

`scripts/runpod_read.py` reads a whole document aloud. It strips Markdown to
speakable prose, splits it at sentence boundaries under a word budget, clones
each chunk through the same endpoint with one fixed seed, and joins the chunk
WAVs with a silence gap that widens at paragraph breaks. The result is one
24 kHz mono 16-bit WAV.

```bash
# One section of a document — the heading names it.
python scripts/runpod_read.py --endpoint-id 53bev6svysh8g4 --voice bob \
  --input path/to/video-intro.md --section "## Script" \
  --output outputs/hyperdev_delegation_intro_bob.wav \
  --work-dir outputs/reads/hyperdev_delegation_intro_bob

# A whole article, with an MP3 alongside.
python scripts/runpod_read.py --endpoint-id 53bev6svysh8g4 --voice bob \
  --input path/to/final.md \
  --output outputs/hyperdev_delegation_article_bob.wav \
  --work-dir outputs/reads/hyperdev_delegation_article_bob --mp3
```

Frontmatter, images, code blocks, HTML blocks, link URLs, footnote markers and
bodies, and a closing italic author footer all come out; heading text, link
text and table cells stay. Nothing else is expanded or rewritten, so a symbol
the author wrote for the eye — an arrow in `idea -> communication -> outcome`,
say — reaches the model as written.

Chunks default to a 70-word budget with a 110-word hard cap, which is about
44 s of speech against the roughly 115 s the token ceiling allows. Only a
single sentence longer than the cap is ever cut, at clause boundaries. When the
worker still answers `truncated: true`, the reader halves that chunk at a
sentence boundary and retries, up to three times.

The work directory holds one WAV per request plus `manifest.json`, which
records each chunk's index, text, SHA, duration, `executionTime` and
`truncated` flag. A re-run resynthesises only chunks whose text changed or
whose WAV is missing, so an interrupted reading resumes where it stopped.
`--mp3` transcodes through ffmpeg and says so and skips when ffmpeg is absent.

Useful knobs: `--word-budget` and `--max-words` for chunk size,
`--sentence-gap-ms` (350) and `--paragraph-gap-ms` (700) for the joins,
`--seed` for reproducibility, and `--rate-per-second` for the cost line, which
defaults to the RTX 4090 tier.

### Two readings on record

Both against endpoint `53bev6svysh8g4` with the registered `bob` voice, seed 42
and the shipped defaults. No chunk needed re-splitting in either.

| Reading | Chunks | Audio | Endpoint execute | Balance delta |
|---|---|---|---|---|
| `## Script` of the HyperDev video intro | 10 | 114.96 s | 242.2 s | $0.041 |
| The HyperDev delegation article, whole | 72 | 977.88 s | 2006.7 s | $0.751 |

Generation ran at 2.16 s of worker time per second of speech, slower than the
single-request figure above because every chunk pays its own prepare step. The
model reads at about 165 words per minute, so a 110-word chunk is roughly 40 s
of speech.

Wall time is longer than execution time: 401.7 s and 2180.2 s respectively, for
a cold start on the first request plus the 2 s `/status` poll interval on every
one after it.

The balance delta runs above the printed cost estimate — $0.751 against $0.614
for the article — because RunPod bills worker runtime, and a worker stays up
through the `idleTimeout` between sequential requests. Read the estimate as the
generation floor, not the invoice. The deltas also carry the network volume's
continuous charge and anything else on the account at the time.

## OpenAI-compatible route

`scripts/openai_shim.py` is a small Starlette app exposing
`POST /v1/audio/speech` in OpenAI's shape, so any tool that lets you override
the OpenAI base URL can drive this endpoint. It loads no model and talks HTTP to
RunPod, so it runs anywhere.

```bash
SHIM_API_KEY=<token you invent> RUNPOD_ENDPOINT_ID=<endpoint id> \
  bash scripts/run_shim.sh          # http://127.0.0.1:8080
```

`RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` come from the environment or
`.env.local`. `SHIM_API_KEY` is the bearer token callers must present; the shim
refuses to serve without one configured, since it is meant to sit on a network.
`docker/Dockerfile.shim` builds it on `python:3.12-slim` with ffmpeg, a few tens
of megabytes rather than several gigabytes.

Request mapping:

| OpenAI field | Handling |
| --- | --- |
| `model` | Accepted and ignored — this server has one model |
| `input` | The text to speak, capped at 5000 characters |
| `voice` | A name in the registry above |
| `response_format` | `wav` and `mp3`; default `mp3` per OpenAI. `opus`, `aac`, `flac` and `pcm` are refused rather than silently served as something else |
| `speed` | Rejected with 400 unless it is exactly 1.0 |

`speed` is refused because nothing in the generation path takes a rate or
duration parameter — `FastStreamingConfig` (`models/fast_streaming.py`) has no
such field — so accepting it would mean ignoring it. MP3 comes from ffmpeg,
which the image already installs; no maintained pure-python MP3 *encoder*
exists, only decoders. Inputs over 600 characters are queued through `/run` and
polled, since a cold worker can outlast the `/runsync` window.

```bash
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H "Authorization: Bearer $SHIM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Hello there.","voice":"bob","response_format":"wav"}' \
  --output hello.wav
```

### Pointing Open WebUI at it

Settings → Audio → Text-to-Speech:

| Field | Value |
| --- | --- |
| Text-to-Speech Engine | OpenAI |
| API Base URL | `http://<shim host>:8080/v1` |
| API Key | the `SHIM_API_KEY` value |
| TTS Model | `tts-1` (any string; it is ignored) |
| TTS Voice | `bob`, or any registered name |

Open WebUI requests `mp3` by default, which the shim serves. Register a voice
before setting this up: the shim has no upload path, by design — a voice is
registered once through `scripts/runpod_clone.py`.

## Live test on record

Two calls against endpoint `53bev6svysh8g4`, both `COMPLETED`:

- 40 s reference — `outputs/Recording_1_ref40.wav` with its transcript from
  `outputs/Recording_1_ref_transcripts.json`, via `/run` plus `/status` polling.
  Result: `outputs/voice_clone_serverless_Recording_1_40s_long.wav`, 11.04 s at
  24 kHz. Text matches `outputs/voice_clone_real_Recording_1_40s_long.wav` from
  the pod so the two are directly comparable.
- 20 s reference — `outputs/Recording_1_ref20.wav`, via `/runsync`. Result:
  `outputs/voice_clone_serverless_Recording_1_20s.wav`, 4.40 s at 24 kHz. The
  repository holds no transcript for this clip, so the reference text is the
  prefix of the 40 s transcript that covers what the 20 s clip contains,
  cross-checked against a local faster-whisper pass.
