"""Import-level validation used while building the public container."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import qwen_tts
import torch
import transformers

from models.fast_streaming import FastStreamingConfig
from models.warmup_profile import load_warmup_profile

EXPECTED = {
    "torch": "2.9.1",
    "transformers": "4.57.3",
    "qwen-tts": "0.1.1",
}

# The serverless image is built with BUILD_FLASH_ATTN=0, so flash-attn is
# optional here. Every load_runtime caller hardcodes attn_implementation="eager"
# (infer.py:73, breeze_infer/api.py:131), so its absence changes no code path.
OPTIONAL = {"flash-attn": "2.8.3"}


def main() -> None:
    imported_modules = [qwen_tts, torch, transformers]
    expected = dict(EXPECTED)

    for name, version in OPTIONAL.items():
        module_name = name.replace("-", "_")
        if importlib.util.find_spec(module_name) is None:
            print(f"optional dependency absent, skipping: {name}")
            continue
        imported_modules.append(importlib.import_module(module_name))
        expected[name] = version

    if not all(imported_modules):
        raise RuntimeError("one or more required modules failed to import")
    versions = {name: importlib.metadata.version(name) for name in expected}
    for name, want in expected.items():
        actual = versions[name].split("+")[0]
        if actual != want:
            raise RuntimeError(f"{name}: expected {want}, got {versions[name]}")

    cfg = FastStreamingConfig(fast_all=True)
    if not cfg.fast_all:
        raise RuntimeError("fast runtime configuration is unavailable")

    load_warmup_profile(REPO_ROOT / "configs/fast.json")

    print("Container dependency smoke check passed:", versions)


if __name__ == "__main__":
    main()
