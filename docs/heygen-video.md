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

An avatar scene carries `input.avatar_id`, `input.audio_asset_id` and, when you
want to override the engine, `input.engine` as an object — `{"type":
"avatar_iii"}`, not a bare string. Leaving `engine` out gets HeyGen's documented
default, Avatar IV, which is what `build_storyboard.py` does unless `--engine`
says otherwise. An image scene carries `source` (`{"type": "asset_id",
"asset_id": …}`) and its own `audio_asset_id` at the scene's top level. There is
no per-scene caption or
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

Resume validates the audio it finds, not just its filename: a re-run re-reads
every part WAV in the work directory and re-synthesises any chunk whose file is
empty, truncated, or a different length from the duration the manifest recorded.
An interrupted reading therefore cannot come back one chunk short.

Build the storyboard from the intro audio, the body work directory, the closing
audio and the images.

```bash
python scripts/build_storyboard.py \
  --intro-audio outputs/hyperdev_test/audio/intro.wav \
  --body-dir outputs/hyperdev_test/audio/body \
  --closing-audio outputs/hyperdev_test/audio/closing.wav \
  --image .../images/hero.png --image .../images/body-1-soloist.png \
  --image .../images/body-2-score.png --image .../images/body-3-rehearsal-hall.png \
  --look-id b112b52a65e74f89aee15343df6ac0a7 --captions \
  --title "Delegation as an Engineering Skill — HyperDev test" \
  --output outputs/hyperdev_test/storyboard.json
```

Pick the look with `GET /v3/avatars/groups/{group_id}/looks` and read
`image_width`, `image_height`, `preferred_orientation` and
`supported_api_engines` off each entry. A 1920×1080 landscape look needs no
crop at 16:9 1080p; a square or portrait look would be centre-cropped to the
canvas.

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

Voice `bob`, seed 42, endpoint `53bev6svysh8g4`, output 16:9 1080p. The
on-camera scenes use look `b112b52a65e74f89aee15343df6ac0a7`, "Focused software
developer at desk", from the "Masa" group `cd2d487982134b85887f2b13459d9886`:
a `photo_avatar`, 1920×1080 landscape, `status: completed`, so it fills a 16:9
1080p frame with no crop. All 16 looks in that group support `avatar_iii`,
`avatar_iv` and `avatar_v`; the storyboard names none, so HeyGen's default —
Avatar IV — applies, and `expressiveness` defaults to `low`. A photo avatar
needs no consent step, and none was required: every render went through on the
first submission.

An earlier pass rendered the same storyboard against the "Robert Matsuoka"
digital twin `1a8b4f7828bd4b7fb778e06e92d87460` on `avatar_iii`. Both runs are
recorded below, because the pair is the only per-engine cost comparison this
account has.

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
add-on credits before any render.

| Render | Avatar, engine | Video | Video id | Premium credits after | `api` quota after |
|---|---|---|---|---|---|
| Closing only | digital twin, `avatar_iii` | 13.12 s | `3c2ea3227dde51c5d7087fde64f1f625` | 21 | 4532 |
| Full, 11 scenes | digital twin, `avatar_iii` | 295.04 s | `474e593f17199cb569c3dd651c21c6c0` | 21 | 4288 |
| Closing only | photo avatar, default | 13.12 s | `3e819a2b219a89eb8623d67dab522b91` | 21 | 4257 |
| Full, 11 scenes | photo avatar, default | 295.04 s | `829b77d07de87d4936788e5a2ddc9568` | 21 | 3824 |

**The premium-credit counter never moved.** Four renders, 616 seconds of video,
and `premium_credits.remaining` stayed at 21 with `add_on_credits.remaining` at
1,505. API renders on this account draw on an API quota instead, which is the
column that does move.

That quota is what makes a per-second cost measurable. The photo-avatar probe
cost 31 units for 13.12 s of avatar, 2.36 units per second. Projecting the whole
295-second video at that rate gives about 700 units against 4,257 remaining, and
zero premium credits against the five the budget required, so the full render
went ahead. It actually cost 433 units. The gap is the image scenes: 128.08 s of
avatar at 2.36 units/s is 302 units, leaving 131 units for 166.96 s of image
scenes, about 0.78 units per second. A still image over a voiceover is roughly a
third the price of an animated face.

The digital-twin pass on `avatar_iii` cost 244 units for the same 295 seconds,
against 433 for the photo avatar on the default engine — the photo-avatar route
is about 1.8× the price here.

Read those unit figures as this account's own measurements, not as documented
pricing. The counter comes from `GET /v2/user/remaining_quota`, which is
deprecated (sunset 2026-10-31) and does not document its unit; `GET /v3/users/me`
is the supported endpoint and reports only the integer credit pools, which are
too coarse to show any of this.

### Render times

| Render | Video length | Server time |
|---|---|---|
| Closing, digital twin | 13.12 s | 40 s |
| Full, digital twin | 295.04 s | 300 s |
| Closing, photo avatar | 13.12 s | 43 s |
| Full, photo avatar | 295.04 s | 200 s |

Roughly real time or better. A 15-minute polling budget is generous for a video
of this length; `status --timeout` defaults to 1800 s.
