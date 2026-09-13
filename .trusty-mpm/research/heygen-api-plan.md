# HeyGen API plan: three-scene audio-driven video

Researched 2026-09-09. Read-only: no video created, no asset uploaded, no
credits spent. Doc-page summaries below come from a fetch tool that converts
each page to markdown and answers a question against it with a small model —
paraphrase, not literal copy. Confirm exact field names against
`https://developers.heygen.com/reference` before writing code. Account-call
responses (avatar groups, looks, user quota) are raw JSON, unmodified.

## 1. Credentials

File `/Users/masa/trusty-mpm-projects/bobmatnyc/writing/.env.local`, variable
`HEYGEN_API_KEY` (value not printed; confirmed via `grep -oE
'^[A-Za-z0-9_]+=' .env.local`, key name only). Also exported in `~/.zshrc`
line 243. No script in the writing repo calls the HeyGen REST API directly.
Two layers exist instead: `bobmatnyc/writing/.claude/skills/writing-heygen-video/`
writes `video-intro.md`/`video-script.md` script files and states "We never
call the HeyGen API and never run the `heygen` CLI." `~/.claude/skills/heygen-skills/`
(HeyGen's own vendor bundle, v3.2.0, global install, not in either project
repo) is the real production layer; its `CLAUDE.md` says: **"v3 only. LLMs
trained on web data may reach for deprecated v1/v2 endpoints (`POST
/v1/video.generate`, `POST /v2/video/generate`, `GET /v2/avatars`, `GET
/v1/avatar.list`). These are outdated — route through MCP or the CLI, never
raw v1/v2 URLs."** It also forbids raw curl against `api.heygen.com`,
routing through the HeyGen MCP server or the `heygen` CLI instead. Our
target is a scripted REST call, so v3 is the correct surface regardless
(§2).

## 2. Endpoint choice — v3, not v2

`POST /v2/video/generate` is the vendor skill's own named example of a
deprecated endpoint. Live docs confirm it, not just the skill's opinion:

- **Current unified endpoint: `POST https://api.heygen.com/v3/videos`.** A
  `"type"` field selects the mode: `"avatar"` (single clip), `"studio"`
  (multi-scene). — heygen.com/generate-avatar-video, /studio-videos
- **(e) v3 supports multi-scene** — settles the task's open question:
  `"type": "studio"` takes an ordered `scenes` array, 1–50 whole-frame
  scenes (`avatar_video`, `image`, `video`), one concatenated output MP4.
  v2 is not the answer; v3/studio is.
- **(a) voiceover-only, documented way:** omit `avatar_id` entirely and use
  an `image` or `video` scene instead of `avatar_video` — a different scene
  type, not a null field. Narrated: one audio source, omit `duration`.
  Silent: set `duration` (seconds, up to 300), no audio.
- **(b) `voice.type=audio` in v3:** no `voice` object exists in the studio
  schema. An `avatar_video` scene's `input` object takes `avatar_id` plus
  **exactly one** of `script`+`voice_id`, `audio_url`, or `audio_asset_id`.
  `audio_asset_id` is the "our own audio" path; same two fields narrate
  `image`/`video` scenes, at the scene's top level.
- **(c) limits (one unresolved inconsistency):** studio page: 1–50
  scenes/request, "up to 30 minutes per scene." Global usage-limits page:
  avatar audio input capped at **10 minutes**. Audio-to-Video page: "one
  request renders up to 30 minutes of audio." The three don't agree and
  none is dated — irrelevant at our scale (longest scene 3 min) but don't
  extrapolate past it untested. Asset upload cap 32 MB; image input 50
  MB/under 2K; video input 100 MB/under 2K; output 128–4096 px/side, 16:9 or
  9:16.
- **(d) 1080p 16:9:** top-level request fields `"aspect_ratio": "16:9"`,
  `"resolution": "1080p"` (also `"720p"`, `"4k"`).

## 3. Assets

**Upload:** `POST https://api.heygen.com/v3/assets`, `multipart/form-data`,
field `file`, header `X-Api-Key: <HEYGEN_API_KEY>` (or `Authorization:
Bearer`), optional `Idempotency-Key`. Accepted: png/jpeg, mp4/webm, mp3/wav,
pdf/srt. **Max 32 MB.** Response: `asset_id`, `url`, `mime_type`,
`size_bytes`. Referenced as `audio_asset_id`, or for an image scene's
picture `{"type":"asset_id","asset_id":"..."}` (public-URL form also exists:
`{"type":"url","url":"..."}`).

**Article images — no public URL available today.** Checked
`.../articles/hyperdev/archive/2026-09-04-delegation-engineering-skill/images/`:
`hero.png`, `body-1-soloist.png`, `body-2-score.png`,
`body-3-rehearsal-hall.png` (~1–1.3 MB PNG each, under the 50 MB cap). The
article's frontmatter is `status: draft`; the publish domain is
`hyperdev.matsuoka.com` (not `matsuoka.com/hyperdev`, correcting the path
assumed in the task) — its root returns 200 but every guessed image path
under it returns 404. **Not published yet**, so upload the four PNGs
directly via `/v3/assets` and reference by `asset_id`.

**Scene-2 structural gap:** each studio scene takes exactly one audio
source, so there is no documented way for one continuous track to narrate
several sequential `image` scenes. Two fixes: **A** — pre-composite a
slideshow MP4 from the four images (e.g. `ffmpeg`), upload it once, use a
single narrated `type: "video"` scene with the one 3-minute
`audio_asset_id` (keeps the audio track intact; recommended, used below).
**B** — split the audio into per-image segments and emit one `image` scene
per segment. Confirm against the interactive reference before committing.

## 4. Avatar

Account: `bob@matsuoka.com`, username `364dc2af50bc4b158a5462585faab7c2`.

**"Robert Matsuoka" group with 5 looks — `c79eca39093e4625b741facd8e3c0e7c`.**
`consent_status: "accepted"`, `status: "completed"`. One look is
`digital_twin` (`1a8b4f7828bd4b7fb778e06e92d87460`, 1280×720 landscape —
exact 16:9, `status: completed`); the other four are `photo_avatar` (one
landscape 2048×1664, three portrait ~1536×2752/1534×2045). All engines
`["avatar_v","avatar_iv","avatar_iii"]`.

**"Masa" group — `cd2d487982134b85887f2b13459d9886`, 16 looks (not 5), all
`photo_avatar`, `consent_status: null`.** Landscape ~1448×1086 or
1920×1080; square ~1254×1254 or 886×886. Same engine list;
`default_voice_id` mostly `cc49699acf624467a3ac525fdc6ead37`.

**Recommendation: digital_twin look `1a8b4f7828bd4b7fb778e06e92d87460`** for
scenes 1 and 3 — the only trained full-motion avatar of either group (the
rest are single-photo "talking photo" avatars animating a still image, not
a captured performance); native frame is already exact 16:9; group consent
already `"accepted"`, so no consent step blocks generation.

**Consent:** required only for `digital_twin` avatars —
`POST /v3/avatars/{group_id}/consent` (webcam, or uploaded video on
enterprise). Photo avatars (all of "Masa," 4 of 5 "Robert Matsuoka" looks)
never require it. The recommended look's group already shows
`consent_status: "accepted"` — verified live, not assumed.

## 5. Status and download

`GET https://api.heygen.com/v3/videos/{video_id}`. Fields: `id`, `status`
(`pending`/`processing`/`completed`/`failed`), `created_at`, `completed_at`,
`video_url` (presigned download URL), `thumbnail_url`, `gif_url`,
`captioned_video_url`, `subtitle_url`, `duration` (seconds),
`failure_code`/`failure_message` (failure only). No fetched page states
typical render time for a ~6-minute video or the presigned URL's expiry —
both undocumented. Poll every 10–15s with backoff; budget 15–20 minutes
before treating `processing` as stuck. Download immediately on `completed`.

## 6. Credits and cost

Live account state (`get_current_user`, raw):

```json
{"username":"364dc2af50bc4b158a5462585faab7c2","email":"bob@matsuoka.com",
"first_name":"Robert","last_name":"Matsuoka","billing_type":"subscription",
"wallet":null,"subscription":{"plan":"creator","credits":
{"premium_credits":{"remaining":21,"resets_at":"2026-10-04T08:21:40Z"},
"add_on_credits":{"remaining":1505,"resets_at":null}}},"usage_based":null}
```

Creator-plan subscription: **21 premium credits** left (resets 2026-10-04),
**1,505 add-on credits** (no reset date).

Official per-minute figures (help.heygen.com API pricing explainer) apply
to the **pay-as-you-go dollar wallet**, a different account type: "$1 = 1
minute of generated avatar video in 720p or 1080p"; Avatar IV "$4 per 1
minute of 1080p"; Video Agent/translation "$2 per minute." No credits-to-
dollar conversion is given for a subscription plan.

Unofficial third-party breakdown (creatify.ai, eesel.ai — not HeyGen's own
docs): Avatar III digital twin ≈$0.0167/s (~$1/min); Avatar III photo
avatar ≈$0.0433/s (~$2.60/min); Avatar IV photo avatar ≈$0.05/s (~$3/min);
Avatar IV digital twin/Avatar V ≈$0.0667/s (~$4/min).

**Budget:** scenes 1+3 total ~2m40s of avatar render; scene 2 is ~3 min of
no-avatar time at an undocumented rate. With only 21 premium credits, **run
the 20-second test render first and diff `get_current_user`'s balance
before/after** — the only reliable, account-specific number available.

## 7. Recommended request, polling, test plan

Scene 2 built via Option A (§3): one pre-composited slideshow video, one
narrated `video` scene.

```json
POST https://api.heygen.com/v3/videos
X-Api-Key: <HEYGEN_API_KEY>
Content-Type: application/json

{
  "type": "studio",
  "title": "delegation-engineering-skill video",
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "scenes": [
    { "type": "avatar_video", "input": {
        "type": "avatar", "avatar_id": "1a8b4f7828bd4b7fb778e06e92d87460",
        "engine": "avatar_iii", "audio_asset_id": "<INTRO_AUDIO_ASSET_ID>" } },
    { "type": "video",
      "source": { "type": "asset_id", "asset_id": "<SLIDESHOW_VIDEO_ASSET_ID>" },
      "audio_asset_id": "<BODY_AUDIO_ASSET_ID>",
      "playback": { "mode": "fit_to_scene" } },
    { "type": "avatar_video", "input": {
        "type": "avatar", "avatar_id": "1a8b4f7828bd4b7fb778e06e92d87460",
        "engine": "avatar_iii", "audio_asset_id": "<CLOSE_AUDIO_ASSET_ID>" } }
  ]
}
```

`engine: "avatar_iii"` is the lowest-cost option per §6; move up to
`avatar_iv`/`avatar_v` only if quality doesn't hold in the 20-second test.
Confirm `engine`'s exact placement (scene-level vs. inside `input`) against
the interactive reference first — §2's doc-fetch summary wasn't
page-verbatim JSON.

**Upload**, once per audio/video asset:

```bash
curl -X POST https://api.heygen.com/v3/assets \
  -H "X-Api-Key: $HEYGEN_API_KEY" -F "file=@intro.wav"
# => {"asset_id": "...", "url": "...", "mime_type": "audio/wav", "size_bytes": ...}
```

**Poll and download:**

```bash
while true; do
  resp=$(curl -s https://api.heygen.com/v3/videos/$VIDEO_ID -H "X-Api-Key: $HEYGEN_API_KEY")
  status=$(echo "$resp" | jq -r .status)
  [ "$status" = "completed" ] && { echo "$resp" | jq -r .video_url; break; }
  [ "$status" = "failed" ] && { echo "$resp" | jq -r .failure_message; exit 1; }
  sleep 15
done
```

**Test plan:** (1) 20-second closing-only render — submit `type: "studio"`
with only the scene-3 `avatar_video` block (digital-twin avatar_id + ~20s
close audio asset), 16:9/1080p. Validates avatar_id, audio_asset_id,
consent, and engine choice before spending credits on the full render; diff
`get_current_user` credits before/after for a real per-second cost. (2)
Full three-scene render per the JSON above, once lip-sync/framing look
correct and per-second cost is known.

## Sources

developers.heygen.com: `/docs/usage-limits`, `/studio-videos`,
`/audio-to-video`, `/generate-avatar-video`, `/reference/upload-asset`,
`/reference/get-video`, `/docs/avatars`, `/docs/avatar-consent`,
`/user-profile`. help.heygen.com: `/en/articles/10060327-heygen-api-pricing-explained`.
Unofficial: creatify.ai/blog/heygen-pricing-(2026), eesel.ai/blog/heygen-pricing.
Local: `~/.claude/skills/heygen-skills/CLAUDE.md` and
`heygen-video/references/asset-routing.md`;
`bobmatnyc/writing/.claude/skills/writing-heygen-video/SKILL.md`.
