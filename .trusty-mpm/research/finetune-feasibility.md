# Breeze TTS 2 — Single-Speaker Fine-Tune Feasibility

## 1. Architecture

Three components, ~3B params total (HF model card, huggingface.co/BreezeBlue/breeze-tts-2):

- **Backbone** — `BreezeBackboneModel` (`models/breeze.py:807`), a Llama-style causal decoder. Default config (`models/breeze_base_config.py:224-243`, "breeze-1b") is hidden_size=2048, 16 layers, 32 heads / 8 KV heads, intermediate=8192, text_vocab=128256, rope_theta=500000 with an `llama3` rope-scaling profile — this is literally Llama-3.2-1B's published config shape, not T5Gemma2. ~1.2B params.
- **Depth decoder** — `BreezeDepthDecoderForCausalLM` (`models/breeze.py:642`), a small 4-layer, hidden_size=1024 transformer that autoregressively predicts each of 32 codebook tokens per audio frame (`models/breeze_base_config.py:23-60`).
- **Text encoder** — `T5Gemma2TextEncoder` (`models/t5gemma2_compat.py:14,589`), a compatibility shim (not upstream `transformers`) with default hidden_size=2304, intermediate=9216, 26 layers — matches Gemma-2-2B's shape. It encodes the natural-language instruction for Voice Design/Direction and is frozen by default: `requires_grad = getattr(config.text_encoder_config, "requires_grad", False)` (`models/breeze.py:1061-1077`).
- **Codec** — bundled `Qwen3TTSTokenizer` (`breeze_infer/runtime.py:97-106`), always run in eval with gradients disabled: `self.codec_model.eval()` / `param.requires_grad = False` (`models/breeze.py:941-945`, re-asserted `1139-1140`).

`models/breeze_base_config.py:1` carries `# Copyright 2025 Sesame and The HuggingFace Inc. team` — the backbone/depth-decoder split and config field names match Sesame AI's CSM-1B architecture (upstreamed into `transformers` as `Csm`), not Qwen3-TTS or CosyVoice. Only the **audio tokenizer** is swapped in from Qwen3-TTS: "The audio tokenizer is based on Qwen3-TTS by the Alibaba Qwen Team" (README.md:184, github.com/QwenLM/Qwen3-TTS, Apache-2.0). So: architecture = Sesame CSM fork + Qwen3-TTS codec + an added T5Gemma2-shaped instruction text encoder for voice design/direction. No CSM or T5Gemma2 training recipe ships in this repo; CSM's own training code (github.com/SesameAILabs/csm) is a plausible starting reference for the backbone+depth-decoder loss shape, though Breeze's checkpoint and tokenizer have diverged from it.

A speaker fine-tune would train the **backbone** and **depth decoder** (the two components with real trainable capacity that jointly predict audio tokens); the codec is hard-frozen and the text encoder is frozen by default and orthogonal to speaker identity.

## 2. Published training code

- **Official**: none. `github.com/breezeblue-ai/breeze-tts` ships inference only (`infer.py`, `breeze_infer/`) — no `train.py`, no `training/` directory, no LoRA config anywhere in this checkout (`grep -rn "train\|LoRA" **/*.py` under the repo root turns up only `model.training` dropout-mode checks and `--fast-*` inference flags). The repo's open issues contain no fine-tuning discussion (github.com/breezeblue-ai/breeze-tts/issues, filtered search returned zero matches as of 2026-09-08). The HF model card documents Voice Clone/Design/Direction as reference-conditioned zero-shot inference only, no fine-tuning workflow.
- **Community, partial**: `github.com/instavar/breeze-tts2-finetuning` — an independent, unofficial toolkit (Apache-2.0 code, 1 star, 11 commits) implementing LoRA and full-SFT training for Breeze TTS 2, with released artifacts `huggingface.co/instavar/sg-narration-lora-r8` (rank-8 LoRA, ~12M trainable params) and `huggingface.co/instavar/sg-narration-full-sft`. Their own writeup (instavar.com/research/tts/breeze-tts-2-lora-full-sft-singapore-english) reports WER 0.047 and speaker-similarity 0.681 on a single consented speaker from Singapore's National Speech Corpus, but explicitly caveats: "one voice profile does not establish performance across ... speakers," evaluation used one blind listener, and confidence intervals on preference gains crossed zero. Treat this as real but thin evidence — a working recipe exists, not a validated one.
- No paper describing a Breeze-specific fine-tuning method was found.

**Verdict: partial.** No official recipe; one small, unofficial, weakly-validated community toolkit exists.

## 3. Licence

- **Code** (this repo, Apache-2.0, `LICENSE`): permits fine-tuning/training code use freely, commercially or not.
- **Model weights and any derivative** (LoRA adapters, merges, distillations, fine-tunes) are separately governed by the BreezeBlue Research and Non-Commercial License (huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE), which README.md:184 also points to. Key terms:
  - Fine-tuning is explicitly contemplated and permitted for **research or non-commercial purposes**: derivatives are defined to include "fine-tune, LoRA, merge, quantization, distillation" (License §1.3), and creating them is allowed under §2.
  - No commercial-use exception of any kind: "does not include a creator, content monetization, small-business, revenue-threshold, or other implied commercial-use exception" (§3).
  - A fine-tuned/derivative model cannot itself be used to train, fine-tune, distill, or evaluate any *non-BreezeBlue* speech/audio/language model (§5(b)) — i.e., no using Breeze outputs to bootstrap a competing model.
  - Redistributing a derivative requires including the license, retaining notices, and attributing "Derived from Breeze TTS 2 ... licensed for research and non-commercial use only" (§4); "BreezeBlue" can't be used as the derivative's primary name.
- Net: personal/research fine-tuning on your own voice is licence-permitted. Any commercial product built on a fine-tuned checkpoint is not, absent BreezeBlue's paid platform terms.

## 4. If using the community recipe

Per the Instavar toolkit's documented approach (not independently verified by running it):
- **Data**: single-speaker audio + exact transcripts, consented; their SG-English run used a National Speech Corpus subset (duration per speaker not disclosed in the fetched summary).
- **Format**: audio/transcript pairs, matching this repo's own `--ref-audio`/`--ref-text` convention (README.md:60-100) of exact-transcript pairing.
- **LoRA**: rank 8, ~12M trainable params, targeting "the semantic backbone and decoder" — consistent with the backbone + depth-decoder fine-tune surface identified in §1.
- **GPU/time**: not stated in the fetched material; given the ~1.2B-param backbone and 12M-param LoRA, a single 24GB+ GPU (the same class this repo already recommends for `--fast-all` inference, README.md:44) doing a few thousand steps over 30-60 min of audio is plausible in low-single-digit GPU-hours, but this is inference from model size, not a number Instavar published — verify before budgeting.
- Full-SFT variant exists (`sg-narration-full-sft`) but is far more expensive and, per their own results, didn't clearly beat LoRA.

## 5. If building your own (fallback / more rigorous path)

The inference model already computes the right losses — `BreezeOutputWithPast` returns `loss`, `depth_decoder_loss`, and `backbone_loss` for next-token / next-codebook prediction (`models/breeze.py:78-116`), and `codec_model.encode(...)` turns raw audio into codebook targets for teacher forcing (`models/breeze.py:1669-1670`). What's missing is the training harness: an optimizer loop, a dataset collator pairing (text, ref-audio-codes) → target codebook sequences, checkpoint/LoRA wiring (e.g. via `peft`), and an eval harness (WER + speaker-similarity, as Instavar used).

- **Approach**: LoRA on the backbone's attention/MLP projections (`BreezeAttention`/`BreezeMLP`, `models/breeze.py:289,189`) and optionally the depth decoder; keep `codec_model` frozen (already enforced, `models/breeze.py:944-945`) and keep the text encoder frozen (already the default, `models/breeze.py:1061`) so instruction-following for Voice Design/Direction isn't disturbed.
- **Effort**: roughly 3-7 engineer-days for someone comfortable with HF `Trainer`/`peft` — this is "wire a training loop onto an existing loss," not "invent audio-token fine-tuning from scratch." The Instavar project is one data point that this is buildable at small-team scale.
- **Risk**: full fine-tuning (vs. LoRA) of the shared backbone risks degrading the CFG-scale instruction-following that Voice Design/Direction depend on (README.md:104-137 documents `--cfg-scale` steering) since the same backbone weights serve both timbre and instruction-conditioning; a low LR, LoRA-only, few-epoch run is the safer default. Instavar's own confidence intervals crossing zero suggest even a competent implementation may yield marginal gains over zero-shot for one speaker.
- **Alternative requiring no training**: a stored-reference voice registry — keep the single best ~20-40s clean clip and call the existing zero-shot `--ref-audio`/`--ref-text` Voice Clone path (README.md:60-84) at inference time. More total audio helps this approach only indirectly: it gives you more candidate segments to pick the cleanest, most representative one, and lets you keep several reference clips for different prosodic contexts — it does not, by itself, improve one zero-shot call's fidelity, since each inference call conditions on exactly one reference clip. Given the license restricts fine-tunes to non-commercial use anyway and the community evidence for fine-tuning's benefit over zero-shot is weak, the registry approach is the lower-risk default; fine-tuning is a fallback if the best single reference clip's zero-shot output is judged insufficient.

## 6. Inventory — `/Users/masa/Projects/voice-cloning/recordings/`

15 audio files (`.mp3`, each with a matching `.wav` in `wav/`), all with a matching `.txt` transcript alongside, plus `transcription_report.json` (structured, per-file word/segment counts) and `transcription_summary.txt`. Transcript contents were not printed here per instructions.

| File | Duration |
|---|---|
| Recording 1.mp3 | 4:31 (271.1s) |
| unique_bob_002.mp3 | 5:17 (317.1s) |
| unique_bob_001.mp3 | 3:02 (182.4s) |
| unique_bob_049.mp3 | 2:12 (132.1s) |
| bob_voice_0004.mp3 | 2:06 (125.7s) |
| unique_bob_010.mp3 | 1:37 (97.1s) |
| unique_bob_004.mp3 | 1:35 (95.4s) |
| unique_bob_038.mp3 | 1:30 (90.2s) |
| bob_voice_0001.mp3 | 1:13 (72.9s) |
| unique_bob_006.mp3 | 1:13 (73.0s) |
| bob_voice_0017.mp3 | 1:01 (60.7s) |
| unique_bob_048.mp3 | 0:46 (46.1s) |
| bob_voice_0021.mp3 | 0:44 (43.8s) |
| bob_voice_0027.mp3 | 0:39 (39.4s) |
| unique_bob_013.mp3 | 0:36 (35.9s) |

**Total: 28.0 minutes (1,682.7s) across 15 clips** — just under the 30-minute threshold in the question. All clips have transcripts (via `ffprobe`, file listing, and `transcription_report.json`; contents not read into this report).

## 7. Recommendation

1. No official Breeze TTS 2 fine-tuning code exists; only one small, unofficial, weakly-validated community LoRA/full-SFT toolkit (Instavar) does.
2. The model license permits non-commercial fine-tuning and derivative adapters, but forbids commercial use and using Breeze outputs to train other foundation models.
3. You have ~28 minutes, just under 30 — start with the zero-shot path already in this repo: pick/trim the cleanest ~20-40s single-speaker clip and evaluate `--ref-audio`/`--ref-text` output before investing in training.
4. If that's insufficient, LoRA-fine-tune the backbone + depth decoder (freeze codec and text encoder) — a ~3-7 day build reusing this repo's existing loss code, non-commercial use only.
5. Given the community evidence of fine-tuning's benefit over zero-shot is thin (N=1 speaker, one blind listener, mixed significance), treat fine-tuning as a fallback, not the default.
