# Assembling a HeyGen video from a Breeze TTS reading

`scripts/runpod_read.py` turns a document into narration WAVs.
`scripts/build_storyboard.py` arranges those WAVs into scenes, and
`scripts/heygen_video.py` uploads them and drives HeyGen's v3 API to a finished
MP4. The three run in sequence: reader, storyboard, render, status, download.

The shape of the result is fixed: an avatar scene reading the intro, one
still-image scene per body chunk with that chunk's audio over it, and an avatar
scene reading the close. The images cycle in the order given, hero first.

## The API surface

Everything goes through v3. `POST /v2/video/generate` is deprecated and is not
used here.

| Call | Purpose |
|---|---|
| `POST /v3/assets` | `multipart/form-data`, one `file` field, 32 MB cap. Returns `data.asset_id`. |
| `POST /v3/videos` | `type: "studio"` plus an ordered `scenes` array, 1 to 50 scenes. Returns `data.video_id`. |
| `GET /v3/videos/{id}` | `data.status` is `pending`, `processing`, `completed` or `failed`, with `video_url` and `thumbnail_url` on success and `failure_message` on failure. |
| `GET /v3/users/me` | Credit balance. The MCP server's `get_current_user` reads the same thing. |

An avatar scene carries `input.avatar_id`, `input.audio_asset_id` and
`input.engine` as an object — `{"type": "avatar_iii"}`, not a bare string. An
image scene carries `source` (`{"type": "asset_id", "asset_id": …}`) and its own
`audio_asset_id` at the scene's top level. There is no per-scene caption or
text-overlay field in the studio schema, so `build_storyboard.py --captions`
records each scene's spoken text in the storyboard for provenance only; pass
`heygen_video.py render --burn-captions` to ask HeyGen for its own
transcript-derived captions instead.

## The sequence

Read the body. A word budget of 80 gives one chunk per source paragraph, which
is what puts the pauses where the script's paragraph breaks are.

```bash
python scripts/runpod_read.py --endpoint-id 53bev6svysh8g4 --voice bob \
  --input outputs/hyperdev_test/abridged.md \
  --output outputs/hyperdev_test/audio/body.wav \
  --work-dir outputs/hyperdev_test/audio/body \
  --word-budget 80 --seed 42 --env-file .env.local
```

Build the storyboard from the intro audio, the body work directory, the closing
audio and the images.

```bash
python scripts/build_storyboard.py \
  --intro-audio outputs/hyperdev_test/audio/intro.wav \
  --body-dir outputs/hyperdev_test/audio/body \
  --closing-audio outputs/hyperdev_test/audio/closing.wav \
  --image .../images/hero.png --image .../images/body-1-soloist.png \
  --image .../images/body-2-score.png --image .../images/body-3-rehearsal-hall.png \
  --look-id 1a8b4f7828bd4b7fb778e06e92d87460 --engine avatar_iii --captions \
  --title "Delegation as an Engineering Skill — HyperDev test" \
  --output outputs/hyperdev_test/storyboard.json
```

Two avatar scenes plus one per body chunk must stay under HeyGen's 50-scene
ceiling. When the body has more chunks than fit, the builder merges adjacent
chunks — concatenating their WAVs with the reader's own paragraph gap — and says
so on stderr. Nine chunks needed no merging here.

Render, poll, download. Uploads are cached by file sha256, so a re-run of
`render` re-uses every asset id instead of uploading again.

```bash
python scripts/heygen_video.py --cache outputs/hyperdev_test/.heygen-assets.json \
  render outputs/hyperdev_test/storyboard.json \
  --request-out outputs/hyperdev_test/request_full.json   # prints the video id

python scripts/heygen_video.py status <video_id> --timeout 2400 --interval 20
python scripts/heygen_video.py download <video_id> \
  --output outputs/hyperdev_test/delegation_test.mp4 \
  --thumbnail outputs/hyperdev_test/delegation_test.jpg
```

`render --dry-run` prints the request body instead of submitting it. It still
uploads, because an upload costs no credits and the cache makes the later real
run free.

The key comes from `HEYGEN_API_KEY` in the environment, or from `--env-file`
when the environment does not carry it.

## What the test run measured

Voice `bob`, seed 42, endpoint `53bev6svysh8g4`, avatar look
`1a8b4f7828bd4b7fb778e06e92d87460` (the "Robert Matsuoka" digital twin,
1280×720, consent accepted), engine `avatar_iii`, output 16:9 1080p.

| Part | Source | Chunks | Audio |
|---|---|---|---|
| Intro | `intro.md`, 351 words | 10 | 114.96 s |
| Body | `abridged.md`, 477 words | 9 | 166.96 s of speech, 8.72 s to 23.92 s per chunk |
| Closing | `closing.md`, 44 words | 1 | 13.12 s |

The body's chunk lengths are set by the source paragraphs, not by the word
budget: `chunk_text` never joins two paragraphs, and the longest paragraph here
is 76 words, about 24 s at the voice's measured ~176 words per minute. Raising
`--word-budget` past 80 changes nothing for this text.

RunPod spend for the body and closing readings: $20.3960 to $20.2883, a delta of
$0.1077 against the reader's own $0.1189 estimate.

### Credits

The account is a Creator-plan subscription: **21 premium credits** and 1,505
add-on credits before either render.

| Render | Video | Premium credits after | `remaining_quota.api` |
|---|---|---|---|
| Closing only, 13.12 s | `3c2ea3227dde51c5d7087fde64f1f625` | 21 | 4532 |
| Full, 11 scenes | `474e593f17199cb569c3dd651c21c6c0` | 21 | 4288 |

The premium-credit counter never moved. A 13-second render is well under one
credit and the counter is an integer, so the test render's cost is only bounded,
not measured: under half a credit. Scaling that bound to the full ~4.9-minute
video gives at most 11.3 credits, and HeyGen's documented Creator-plan rate of
one credit per minute gives about 5 — both leave more than the five credits the
budget required, which is why the full render went ahead.

The legacy `GET /v2/user/remaining_quota` counter is finer-grained and did move:
4532 to 4288, 244 units, during the full render. That endpoint is deprecated
(sunset 2026-10-31) and its unit is undocumented, so treat it as a cross-check
and not as the number of record.
