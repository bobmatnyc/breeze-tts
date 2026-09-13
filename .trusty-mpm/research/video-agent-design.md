# Article-to-HeyGen-Video Custom Agent — Research and Design

Constraint from the PM: the agent is **project-tier in the writing repo**
(`/Users/masa/trusty-mpm-projects/bobmatnyc/writing`), not user-tier. This
document is written to that constraint throughout.

## 1. xflux

**What it is.** A local, fully offline image/video generator on Apple Silicon
via MFLUX/MLX — "no cloud API, no telemetry" (`bobmatnyc/xflux/README.md:9-11`).
16 models through one registry (`docs/models.md:5-9`), dispatched via
`src/xgen/model_registry.py`. Launcher `xflux` on PATH
(`/Users/masa/.local/bin/xflux`) resolves the project at `/Volumes/Mobile/X`
and forwards to `xgen`.

**Invocation.**
```
xflux image "<prompt>" --model <name> --width W --height H [--steps N] [--guidance G] [--seed N] [--output DIR]
xflux batch prompts.txt --model <name> [--output DIR]   # one prompt per line, same model for all
```
(`README.md:57-90`). `--raw-prompt` bypasses per-model prompt formatting.
Output directory default `/Volumes/Mobile/X/output/YYYY-MM-DD/images/`,
filename `{prompt-slug}_{model}_{seed}_{timestamp}.png` plus a JSON sidecar
(`README.md:33-38`). Also exposed as an **MCP server** (registered user-tier,
11 tools including `generate_image`/`edit_image`, stdio) — a writing-repo
agent could call `generate_image` directly instead of shelling out, if that
MCP server is enabled for the session (memory drawer `5df1fcdd`).

**16:9 / custom size.** `--width`/`--height` are free integers (default
1152×1536, portrait) — nothing restricts to square/portrait, so
`--width 1920 --height 1080` is a supported call shape, but unbenchmarked
(all published timings are 512×512). **Open question: measure a real 16:9
render before relying on it for a multi-scene batch.**

**Timing (M4 Max, 512×512, `docs/models.md`).** `zimage-turbo` ~43 s/image,
~10.5 GB peak (fastest, 4 steps, guidance-free) — default recommendation for
a scene batch. `dev` (20 steps) is markedly slower, ~20–24 GB peak. `qwen`
~38 GB peak. No 1920×1080 timing exists; resolution scaling is unmeasured.

**Style / house style.** There is **no fixed HyperDev house style**.
Photographic/illustration controls (shot size, lighting, color grade, medium)
exist only as prompt-text vocabulary inside xflux's PHOTOGRAPHER/ILLUSTRATOR
persona `SKILL.md` files under `src/xgen/personas/`, not as structured
parameters (memory `41b2dc33`). The delegation-engineering-skill article's
four images (hero + 3 body) were **not made with xflux** — made via the
writing repo's `writing-openrouter-images` skill, `google/gemini-2.5-flash-image`
via OpenRouter, at **1024×768 (4:3)** (`images/hero.png.prompt.txt:1-2`,
confirmed by `file`). The prompt file shows a hand-written per-article visual
language ("Painterly cinematic concert-hall realism... 4:3 aspect ratio") —
style is decided per article, not drawn from a standing preset. A prior
article (fable-51-news) tried xflux `dev` for hero candidates but shipped the
OpenRouter/Gemini image instead (memory `1dd3488d`). **Conclusion: xflux is
not currently wired into the HyperDev pipeline; the new agent would be first
to use it, specifically to close the 16:9 gap the 4:3 pipeline can't cover**
(see §2, Image Prep).

## 2. Writing pipeline hooks

**hyperdev skill**: `/Users/masa/trusty-mpm-projects/bobmatnyc/writing/.claude/skills/hyperdev/SKILL.md`
— HyperDev voice/structure overlay on `base-writing`; not video-specific.

**video-intro.md rules**, `articles/hyperdev/CLAUDE.md` — current canonical
shape is **four** sections (an addendum superseded the original three):
`## Script` (1.5–3 min, 320–380 words), `## Tone and delivery`,
`## On camera` (avatar cues, <150 words), `## Visuals` (HeyGen scene-block
cue sheet) (memory `dfad5dd0`, `1bd8aa45`; workflow table
`articles/hyperdev/CLAUDE.md:439-449`, Phase 9). **Contradicted by user
instruction 2026-09-09** (memory `db91721f`): the user now wants a
**three-part structure — intro (Script) / full body reading / CTA closer
naming two recent articles** — not yet reflected in `CLAUDE.md`, which still
describes only the intro teaser. The project-tier `writing-heygen-video`
skill anticipates a `video-script.md` (6–12 min narrated, avatar bookends) as
the second format, closer to the body/closer need, but its production path
assumes HeyGen's own Video Agent (avatar + HeyGen TTS, via the vendor
`~/.claude/skills/heygen-skills/heygen-video/` skill and CLI), **not**
Breeze-TTS audio via `audioAssetId`. The breeze-tts `heygen_video.py` path
(§3) is what matches the user's actual pipeline (Breeze voice "bob" →
`audioAssetId` → HeyGen Studio API scenes). **The new agent should follow
the breeze-tts Studio-API path, not the vendor Video Agent path** —
`articles/hyperdev/CLAUDE.md` Phase 9 and `writing-heygen-video/SKILL.md`
both need updating for the three-part structure and this production path
once the new agent ships.

**Scene-block cue-sheet format** to reuse verbatim (from the shipped
`video-intro.md`, `.../2026-09-04-delegation-engineering-skill/video-intro.md`):
```
Scene 1: Hook
Visual: Open on the attached image of ... (hero.png) as a title card ...
VO: "Hi, I'm Bob Matsuoka" ... "building with AI."
Duration: ~9s
```
One block per scene: `Scene N: <label>` / `Visual:` / `VO:` (elided first…last
words matching the Script section) / `Duration:` (guidance only, non-binding
on Video Agent generation, but consumed directly by the Studio API path).

**Article directory layout** (`writing/CLAUDE.md:57-70`): `drafts/{slug}/` →
`archive/YYYY-MM-DD-{slug}/` on publish, each holding `draft.md`, `final.md`,
`research/`, `images/`. **No `video/` subdirectory convention exists yet** in
the writing repo's own CLAUDE.md, but the breeze-tts side already writes a
`video/` directory pattern for a *different* article
(`writing/AVATAR-BOB.md`/tripbot article: `video/make-graphics.py`,
`video/draft-3min-v2.mp4`, `video/final.md`, `video/final-request.json` —
memory drawers `b1cf71b8`, `49f9a19d`). **Recommend the new agent standardize
on `{article-dir}/video/`** as the output root, matching that precedent.

**Existing project-tier agents**: `writing/.claude/agents/*.md` — 31 files,
including writing-specific custom agents `copyeditor.md`, `fact-checker.md`,
`proofreader.md`, `writer.md`, `writing-critic.md`, `pangram-editor.md`, plus
many `.disabled` bundled-agent copies. No agent yet resembles a video/media
producer. `~/.trusty-mpm/agents/` (user-tier) holds 6 agents (copyeditor,
dotnet-engineer, pangram-editor, proofreader, writer, writing-critic) — this
is a **separate, duplicate-named user-tier roster**; project-tier wins on
precedence (§4) so this is not a conflict, just worth knowing it exists.

## 3. Existing pipeline pieces (breeze-tts @ main, `e84e85d`)

| Script | Inputs | Outputs | Key flags |
|---|---|---|---|
| `scripts/runpod_read.py` | `--input` doc, `--endpoint-id`, `--voice`, `--section`/`--drop-section` | `--output` WAV(+MP3), resumable `--work-dir` with `manifest.json` | `--word-budget` (chunk size, default splits ~1/paragraph), `--seed 42`, `--cfg-scale 1.0`, `--timeout 900`, `--mp3`, `--env-file` |
| `scripts/build_storyboard.py` | intro/closing WAV, `--body-dir` of chunk WAVs, `--image` (repeatable, hero first) | `storyboard.json` (intro-avatar / per-body-chunk-image / closing-avatar scenes) | `--look-id`, `--captions` (provenance only, no real overlay), `--title`, merges adjacent body chunks past the 50-scene ceiling |
| `scripts/heygen_video.py` | `storyboard.json` | uploaded-asset cache JSON, `video_id`, final MP4+thumbnail | subcommands `render [--dry-run] [--request-out]`, `status <id> [--timeout] [--interval]`, `download <id> --output --thumbnail`; `--cache` sha256-dedupes uploads |
| `docs/heygen-video.md` | — | reference doc | v3 API surface (`/v3/assets`, `/v3/videos`, `GET /v3/videos/{id}`, `GET /v3/users/me`), cost/timing tables reproduced in §5 |
| `~/.trusty-mpm/skills/breeze-voice/SKILL.md` (user-tier) | — | Workflow A–D runbook | wraps all of the above; states absolute-path invocation convention (below) |

**Pronunciation lexicon (in progress, uncommitted, worktree
`.claude/worktrees/agent-abe144da682093adb`, branch
`feat/pronunciation-lexicon`, based on `main@e84e85d`, not yet a real diff
against `main` until committed).** New `scripts/pronunciations.json`
(currently just `{"Matsuoka": "Mah-tsu-oh-ka"}`) and new
`scripts/speech_text.py` (text-shaping helpers pulled out of
`runpod_read.py` to respect the 500-line production cap — see its own
Why/What/Test docstring). `runpod_read.py` gains `--pronunciations PATH`
(default `scripts/pronunciations.json`), `--no-pronunciations` to disable,
and `--drop-section "Related reading"`-style section removal (repeatable),
applied in order Markdown-strip → section-select/drop → pronunciation
respell, before chunking — so a chunk's manifest hash already reflects every
text decision (`speech_text.py:1-19`). **Design implication: the new agent
should pass `--pronunciations` for every reading that includes "Matsuoka"
(every intro/closer) and use `--drop-section` to remove "Related reading"
and any other page-only sections from a body reading.**

## 4. Custom agent conventions (project-tier)

From `tm-agent-architecture`: an agent is **official** only if its name
resolves inside the framework's `agent_source_dir()` (bundled assets or the
`agents/agents/` submodule) and goes through `compose_agent`/`extends:`
composition. Anything living only under `<project>/.claude/agents/` is
**custom** — edited directly, "no rebuild step needed."

From `tm-capabilities` `references/framework.md` (Agent Tier Precedence):
```
1. <project>/.claude/agents   — highest precedence, hand-placed/custom only
2. ~/.trusty-tools/.../claude-config/agents  — bundled, tm-deployed
3. ~/.claude/agents           — operator's own, read-only to tm
```
So the file lives at
**`/Users/masa/trusty-mpm-projects/bobmatnyc/writing/.claude/agents/hyperdev-video.md`**
and wins over any same-named agent in the other two tiers.

**Frontmatter observed on every existing project-tier custom agent in this
repo** (`copyeditor.md`, `fact-checker.md`, `proofreader.md`, `writer.md`,
`writing-critic.md`): only `name` and a trigger-rich `description` are
required; `model: sonnet` / `role: qa` appear optionally. None declares
`extends:` — **project-tier custom agents skip the compose chain and do not
automatically inherit BASE_AGENT/BASE-WRITING**; shared instructions must be
loaded explicitly (`copyeditor.md`: "Invoke the `base-writing` skill before
touching any text").

**Deployment/validation**: none needed beyond creating the file —
`tm-agent-architecture`'s custom-agent example is exactly
`Edit: .claude/agents/my-project-specific-agent.md`. `tm validate` checks
deployed payload against the *bundled* roster (`cli.md:217`), so it will not
flag a custom agent; `tm doctor` and `tm sessions instructions` confirm what
a live session actually sees.

**Skill attachment**: skills have a three-tier deploy list — project
(`<project>/.claude/skills`), operator home, managed config — with
project-custom winning on a name collision. Every HyperDev-adjacent skill
(`hyperdev`, `writing-heygen-video`, `writing-openrouter-images`,
`writing-bob-voice`) is already project-tier under a `writing-<topic>`
naming convention. **The new skill should follow it**: project-tier at
`writing/.claude/skills/writing-hyperdev-video-pipeline/SKILL.md` (or fold
into a revised `writing-heygen-video`, see §5) — **not** a reference from
the user-tier `breeze-voice` skill, which documents itself as reusable
"from any project" infrastructure and should stay thin and project-agnostic;
article-specific orchestration belongs with the other writing-tier skills.

## 5. Design proposal: `hyperdev-video` (project-tier agent + skill)

**Reaching breeze-tts from the writing repo.** Three options: (1) **absolute-
path invocation (recommended)** — call
`python /Users/masa/trusty-mpm-projects/breezeblue-ai/breeze-tts/scripts/<script>.py`
from the writing repo, writing outputs into the writing repo's article
`video/` dir, never into the breeze-tts checkout. Matches two existing
precedents: `breeze-voice/SKILL.md` ("Always invoke scripts by their
absolute path... write outputs under the calling project's own directory")
and `fact-checker.md` (already points at a cross-volume absolute path). Zero
new packaging. (2) Pip-installable package — new infrastructure
(`pyproject.toml`, versioning, install step) for scripts only ever called
over HTTP, unjustified for four thin CLI wrappers. (3) Copy scripts into the
writing repo — drifts the moment breeze-tts fixes a bug (e.g. the in-flight
pronunciation-lexicon branch), duplicated-code minimalism violation.
**Recommendation: (1)**, with the breeze-tts path recorded as one constant
(`BREEZE=/Users/masa/trusty-mpm-projects/breezeblue-ai/breeze-tts`) at the
top of the new skill, as `breeze-voice/SKILL.md` already does.

**Inputs**: an article directory (`drafts/{slug}/` or `archive/...`), must
contain `draft.md`/`final.md` and `images/`.

**Stages**:

**(a) Script generation — LLM reasoning, no new code.** Agent writes/updates
`video-intro.md` per the three-part structure the user actually wants: intro
(existing Script rules), body = full article as speech (reuse
`speech_text.py`'s Markdown-to-speech reduction, not a re-summarized
abridgment — a change from `writing-heygen-video`'s implicit "teaser only"
framing), closer = CTA to "hyperdev dot matsuoka dot com" naming two recent
published articles (from `articles/hyperdev/archive/`, most-recent-first).
Text itself follows `writing-for-the-ear.md` (numbers/URLs/acronyms spelled
for speech); "Matsuoka" is handled downstream by `--pronunciations`, not in
the script text.

**(b) Visuals — one xflux image per storyboard scene.** **Needs new code**:
`scripts/scene_prompts.py` — derives an xflux prompt per scene from the
scene's `Visual:` text plus a per-article house-style paragraph the agent
writes once per article (no standing house style exists, §1) — and an
**xflux batch wrapper** that loops scenes calling
`xflux image --model zimage-turbo --width 1920 --height 1080 --output {article}/video/images/`
(hero scene first, matching `build_storyboard.py`'s hero-first convention),
writing one PNG + `.prompt.txt` sidecar per scene like the OpenRouter path
already does. Direct fix for the 4:3-vs-16:9 mismatch
`writing-heygen-video/SKILL.md` already flags ("Article images are generated
at 4:3. HeyGen outputs 16:9... An image made fresh for the video is
generated 16:9, which skips the problem").

**(c) Audio via the reader.** Direct reuse, no new code: three
`runpod_read.py` calls (intro/body/closer) per `breeze-voice` Workflow C,
each with `--pronunciations $BREEZE/scripts/pronunciations.json`, body call
additionally using `--drop-section "Related reading"`.

**(d) Storyboard + HeyGen render, credit gate.** Direct reuse:
`build_storyboard.py` (images from (b), hero first, `--look-id` from
`AVATAR-BOB.md`) → `heygen_video.py render --dry-run` to show request/cost
→ **explicit user confirmation required before a real render** (existing
credit-gate convention, `breeze-voice/SKILL.md` Workflow C step 4) →
`render` → `status --timeout 2400` → `download`.

**(e) Outputs**, under `{article-dir}/video/` (matching the tripbot
precedent, §2): `video-intro.md`, `storyboard.json`, `images/*.png(+prompt.txt)`,
`audio/{intro,body,closing}.wav`, `.heygen-assets.json`, `final.mp4` +
thumbnail, and a new **cost report** (`video/cost-report.md`, needs new
code) summarizing RunPod spend and the HeyGen quota-unit estimate
(`render --dry-run` plus post-render `GET /v3/users/me` delta) — no
existing script writes a consolidated report.

**What needs new code** (breeze-tts side): `scripts/scene_prompts.py`,
an xflux batch wrapper for 16:9 per-scene generation, optionally
`scripts/video_pipeline.py` as a single orchestrator gluing (a)–(e) plus
the cost-report writer — parallel to the existing
`runpod_read.py`/`build_storyboard.py`/`heygen_video.py` trio, called by
the writing-repo agent via absolute path.

**What the agent does purely by reasoning**: writing the three script
sections (intro/body/closer) in Bob's voice per `hyperdev`/`base-writing`;
choosing/writing the per-article house-style paragraph for image prompts;
writing each scene's Visual description; picking two recent articles for the
CTA; deciding scene count/pacing; writing the final cost-report prose.

**Acceptance criteria per stage**: (a) `video-intro.md` has all four current
sections plus the body/closer additions, word counts in range (320–380 for
intro), passes `writing-for-the-ear.md`'s TTS rules (numbers spelled out,
no naked URLs); (b) one 1920×1080 PNG per scene exists with a
`.prompt.txt` sidecar, hero image is scene 1; (c) three WAV files exist,
`manifest.json` per work-dir shows zero `truncated: true` chunks; (d)
`storyboard.json` scene count ≤ 50, `heygen_video.py status` returns
`completed`; (e) `video/cost-report.md` exists with both RunPod-$ and
HeyGen-unit totals and `final.mp4` plays.

**Cost model**: RunPod — cold call $0.02–0.03, warm <$0.01, reading ~$0.05/min
of finished audio. xflux — local/free, but wall-clock unmeasured at
1920×1080 (§1); budget ~1–2 min/image on `zimage-turbo` pending a real
benchmark. HeyGen — avatar-on-camera ≈2.4 quota units/s, image-over-voiceover
≈0.78–0.8 units/s (measured, not documented pricing); a prior 4m55s video
cost 433 units.

**Open questions for the user**:
1. xflux style — invent a fresh visual theme per article (as OpenRouter does
   today) or should Bob define one standing house style to reuse?
2. Scene length/count — the breeze-tts default (one scene per source
   paragraph, `--word-budget 80`) can produce many scenes for a full body;
   is per-paragraph granularity acceptable, or should scenes be coarser
   (e.g. per H2 section)?
3. May the agent spend HeyGen quota without asking? The existing credit-gate
   convention says no; confirm it still holds for the three-part format,
   since a full-body render costs meaningfully more than the 433-unit
   intro-only test (4m55s).
4. Update `articles/hyperdev/CLAUDE.md` Phase 9 and
   `writing-heygen-video/SKILL.md` in the same change, since both currently
   describe only the older intro-teaser-only, Video-Agent-only shape (§2)?
