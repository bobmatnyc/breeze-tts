# Assembling a HeyGen video from a Breeze TTS reading

`scripts/runpod_read.py` turns a document into narration WAVs, with
`scripts/speech_text.py` deciding which words it says.
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

## Choosing the words: the lexicon and section dropping

Two reader flags decide what the voice says before any audio exists. Both act on
the Markdown, before chunking, so a change to either shows up as a changed chunk
sha and resume re-synthesises exactly the chunks whose text moved.

`--pronunciations <file>` names a JSON map of written form to spoken respelling;
`scripts/pronunciations.json` is the default and ships one entry:

```json
{ "Matsuoka": "Mah-tsu-oh-ka" }
```

Matching is whole-word and case-insensitive, and the replacement carries the
matched text's case, so `matsuoka` in a spoken URL comes out `mah-tsu-oh-ka`
while `Matsuoka` in a sentence comes out `Mah-tsu-oh-ka`. `Matsuokas` does not
match. Keys are tried longest first, so a two-word entry beats a one-word entry
that is only its first word. `--no-pronunciations` reads every word as written.

A respelling is a guess about how the model reads it, so pick one by ear rather
than by rule. Three candidates were synthesised on voice `bob`, seed 42, and
transcribed with local `faster-whisper` (medium.en, `compute_type="float32"`) as
a weak check — Whisper hearing the real surname back means the respelling did not
drift into a different word:

| Respelling in the text | What Whisper heard |
|---|---|
| `Mah-tsu-oh-ka` | "Matsuoka" |
| `Mahtsuoka` | "Machocca" |
| `Mat-soo-oh-ka` | "Matsuoka" |

The run-together form is the one to avoid; the syllable breaks are what keep the
word intact. `Mah-tsu-oh-ka` ships as the default.

The manifest records the lexicon's sha as `lexicon`, so a work directory says
which respellings produced its audio and a re-run reports the change on stderr.

`--drop-section "Related reading"` removes a section and everything under it,
repeatable for more than one. It matches an ATX heading (`## Related reading`)
and also a line that is nothing but bold text (`**Related reading:**`), which is
where an article's trailing link list usually lives — read aloud that list is a
run of titles and taglines with no sentences in it. The reader writes the indices
it actually read into the manifest as `reading`, and `build_storyboard.py` builds
scenes from that subset, so a dropped section's already-paid-for WAVs stay
cached in `chunks` without re-entering the video.

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

For a whole article, keep the default word budget and drop the link list:

```bash
python scripts/runpod_read.py --endpoint-id 53bev6svysh8g4 --voice bob \
  --input .../final.md --drop-section "Related reading" \
  --output outputs/hyperdev_test/audio/body_full.wav \
  --work-dir outputs/reads/hyperdev_delegation_article_bob \
  --seed 42 --env-file .env.local
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

### The scene budget

Two avatar scenes plus the body scenes must stay under HeyGen's 50-scene ceiling.
The builder packs adjacent chunks into one scene until adding the next chunk
would carry the scene past `--scene-seconds` (default 40), concatenating their
WAVs with the reader's own gaps — the wider paragraph gap between chunks that
ended a paragraph. Those gaps count towards the target, so `--scene-seconds`
bounds the rendered scene rather than the speech inside it; on this body that is
about 0.7 s per join, and ignoring it used to push the longest scene past the
target. A chunk is never split, so a chunk longer than the target becomes its own
scene and is the only way a packed scene exceeds it. `--max-scenes` is a hard
backstop that merges the packed scenes further if the count is still too high —
that step can exceed the target, because HeyGen refuses a 51-scene request
outright while a long scene only looks wrong. Both steps report on stderr.

Packing by duration rather than by chunk count is what makes a long body work. A
15-minute article is about 70 chunks, and chunk length follows the source
paragraphs, so splitting into equal counts gives scenes anywhere from 5 s to
60 s. Packing the full article's 71 chunks to a 40 s target gives **31 body
scenes plus the two avatar scenes — 33 in all**, from 19.0 s to 39.8 s, mean
30.3 s. Equal-count grouping of the same chunks into 30 scenes would have ranged
5.4 s to 57.4 s.

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

## What the full-article run measured

The second run replaced the abridged body with the whole article and both avatar
scripts with lexicon-aware re-reads. Voice `bob`, seed 42, endpoint
`53bev6svysh8g4`, look `b112b52a65e74f89aee15343df6ac0a7`, 16:9 1080p.

| Part | Source | Chunks | Audio |
|---|---|---|---|
| Intro | `intro.md`, 351 words | 9 | 117.09 s |
| Body | `final.md` less "Related reading" | 71 | 959.90 s assembled, 911.60 s of speech |
| Closing | `closing_v2.md`, 65 words | 1 | 22.08 s |

The body reused the existing `outputs/reads/hyperdev_delegation_article_bob`
work directory. Its `speech.txt` holds no lexicon key — the article names
Matsuoka only in the frontmatter, the author footer and a link URL, and the
reader already drops all three — so adding the lexicon changed no chunk text and
dropping "Related reading" only shortened the list. All 71 chunks came back
cached: 0 endpoint calls, 0.1 s of wall time, no spend.

That render went out at 32 scenes, before the packer counted the silence it
inserts: three of its 30 body scenes ran past the 40 s target, the longest at
41.1 s. Rebuilt with the gap-aware packer the same body gives 31 scenes, none
over 39.8 s. The figures below are the render that shipped.

### Credit gate before rendering

`GET /v2/user/remaining_quota` reported `api: 3824`. At the first run's measured
rates — 2.4 units per avatar-second, 0.8 per image-second — 139.17 s of avatar
and 939.95 s of images project to 1,086 units, leaving about 2,738 against a
500-unit floor. The gate passed and the render went ahead.

### What it cost

| Measure | Before | After | Delta |
|---|---|---|---|
| HeyGen `api` quota | 3,824 | 2,562 | 1,262 units |
| RunPod balance | $20.2051 | $20.0235 | $0.1816 |

The render finished in 334 s for 1,079.13 s of video, about 3.2× real time, and
`ffprobe` reports 1920×1080 h264 at 25 fps.

The projection was 16% low: 1,086 units predicted against 1,262 spent. Holding
the avatar rate at 2.4 units/s, the image scenes came out at about 0.99 units per
second rather than the first run's 0.78. Project image scenes at 1.0 units/s
until a third run says otherwise; the floor check is what makes an under-estimate
survivable, not the estimate itself.

RunPod spend covers three one-sentence pronunciation probes, the intro re-read
and the closing read; the body cost nothing because every chunk was cached. The
reader's own estimate for that work was $0.121 against the $0.182 the balance
moved. The gap is worker delay time — the first probe waited 355 s for a cold
start — which RunPod bills and `executionTime` does not report, plus an
unrelated pod on the account spending $0.023/hr throughout.

## What the first test run measured

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
| Full article, 32 scenes | photo avatar, default | 1079.13 s | `6f501fa9e84e7d30f19286bf03023215` | 21 | 2562 |

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
| Full article, photo avatar | 1079.13 s | 334 s |

Faster than real time in every case, and the margin widens with length: the
18-minute video rendered in 3.2× real time against the 295-second video's 1.5×.
`status --timeout` defaults to 1800 s, which is enough for both.
