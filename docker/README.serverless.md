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

`docker/build.sh` and `docker/run.sh` are unchanged — the defaults keep the
pod/dev image byte-identical to before.

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

The worker looks for `/runpod-volume/breeze-tts-2/audio_tokenizer` — the
directory `load_runtime` hard-fails without (`breeze_infer/runtime.py:99-105`).
If it is missing, the worker downloads `BreezeBlue/breeze-tts-2` from Hugging
Face into the volume before loading, so the first cold start seeds itself and no
separate seeding pod is needed. The repository is public and ungated; no
`HF_TOKEN` is required. Set `BREEZE_CHECKPOINT_DIR` to load from somewhere else.

## Invoking it

```bash
python scripts/runpod_clone.py \
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
// input
{ "text": "...", "ref_audio_b64": "<base64 WAV>", "ref_text": "...",
  "seed": 42, "cfg_scale": 1.0 }

// output
{ "audio_b64": "<base64 WAV>", "sample_rate": 24000,
  "audio_seconds": 4.4, "prepare_seconds": 0.124,
  "generation_seconds": 13.046, "model_load_seconds": 4.664 }
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
