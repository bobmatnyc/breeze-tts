#!/usr/bin/env python3
"""Drive HeyGen's v3 video API: upload assets, render a studio video, poll, download.

Why: the reader in `scripts/runpod_read.py` produces narration WAVs, but turning
them into a talking-head video means four HTTP conversations with HeyGen —
upload every audio and image file, submit one multi-scene request, wait for the
render, fetch the MP4. Doing that by hand re-uploads the same files on every
retry and loses the API's error body behind a shell pipeline.

What: one client with four subcommands. `upload` pushes a file to
`POST /v3/assets` and remembers the returned id keyed by the file's sha256, so a
re-run costs nothing. `render` reads a storyboard JSON, uploads whatever is not
cached, submits a single `type: "studio"` request and prints the video id.
`status` polls `GET /v3/videos/{id}` to a bounded deadline. `download` writes the
finished MP4, and the thumbnail when asked.

The v3 studio schema has no per-scene caption or text-overlay field, so a
scene's `caption` is carried in the storyboard for provenance only; pass
`--burn-captions` to ask HeyGen for its own transcript-derived captions instead.

Usage:
    python scripts/heygen_video.py render storyboard.json --cache assets.json
    python scripts/heygen_video.py status <video_id> --timeout 1800
    python scripts/heygen_video.py download <video_id> --output out.mp4

Test: `test_build_request_orders_scenes_and_sets_globals`,
`test_upload_reuses_a_cached_asset_id`, `test_render_uploads_then_submits`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

API_ROOT = "https://api.heygen.com/v3"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env.local"
DEFAULT_CACHE = Path(".heygen-assets.json")
# POST /v3/assets refuses anything larger.
MAX_ASSET_BYTES = 32 * 1024 * 1024
CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".pdf": "application/pdf",
    ".srt": "application/x-subrip",
}
SCENE_AUDIO_KEY = "audio_asset_id"
TERMINAL_STATES = frozenset({"completed", "failed"})
# A 429 or a 5xx is worth retrying; a 4xx that is not 429 is a request the
# server will refuse again however long we wait.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 2.0
# Past this, honouring the server's own Retry-After would outlast the caller.
MAX_RETRY_AFTER_S = 120.0

# (method, url, headers, body) in, (status_code, headers, body_bytes) out.
Transport = Callable[
    [str, str, dict[str, str], bytes | None], tuple[int, dict[str, str], bytes]
]


class HeyGenError(RuntimeError):
    """An API call that did not return 2xx, carrying the response body verbatim.

    Why: HeyGen reports the offending field in `error.message`, and losing that
    turns a one-line schema fix into a guessing game.
    """

    def __init__(self, method: str, url: str, status: int, body: bytes) -> None:
        text = body.decode("utf-8", "replace").strip()
        super().__init__(f"{method} {url} -> HTTP {status}: {text}")
        self.status = status
        self.body = text


# --- Transport -------------------------------------------------------------


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> tuple[int, dict[str, str], bytes]:
    """Send one request with the standard library, returning status and bytes.

    An HTTP error is a response, not an exception, because the caller wants the
    error body and its `Retry-After`. Only a transport failure propagates.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()


def auth_headers(api_key: str) -> dict[str, str]:
    """Headers every v3 call needs. A plain user agent keeps proxies happy."""
    return {"X-Api-Key": api_key, "User-Agent": "breeze-tts-heygen/1.0"}


def retry_delay(headers: dict[str, str], attempt: int) -> float:
    """Seconds to wait before the next attempt.

    The server's `Retry-After` wins when it sends one, capped so a wild value
    cannot park the caller; otherwise back off exponentially from the base.

    Test: `test_retry_delay_honours_retry_after`,
    `test_retry_delay_backs_off_exponentially`
    """
    raw = next(
        (value for key, value in headers.items() if key.lower() == "retry-after"), None
    )
    if raw:
        try:
            return min(max(float(raw), 0.0), MAX_RETRY_AFTER_S)
        except ValueError:
            pass  # A HTTP-date Retry-After falls through to the backoff.
    return BACKOFF_BASE_S * (2 ** (attempt - 1))


def call(
    transport: Transport,
    method: str,
    path: str,
    api_key: str,
    *,
    json_body: dict | None = None,
    extra_headers: dict[str, str] | None = None,
    raw_body: bytes | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Make one v3 call and return the `data` object, raising on any non-2xx.

    A 429 or 5xx is retried up to `MAX_ATTEMPTS` times with exponential backoff,
    or after the server's own `Retry-After` when it sends one. Every other
    non-2xx raises on the first response, because waiting will not fix it.

    Test: `test_call_raises_with_the_response_body`,
    `test_call_retries_a_429_then_succeeds`,
    `test_call_gives_up_after_max_attempts`
    """
    url = f"{API_ROOT}{path}"
    headers = auth_headers(api_key)
    body = raw_body
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(json_body).encode("utf-8")
    headers.update(extra_headers or {})
    for attempt in range(1, MAX_ATTEMPTS + 1):
        status, response_headers, payload = transport(method, url, headers, body)
        if 200 <= status < 300:
            parsed = json.loads(payload or b"{}")
            return parsed.get("data", parsed)
        if status not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
            raise HeyGenError(method, url, status, payload)
        delay = retry_delay(response_headers, attempt)
        print(
            f"{method} {url} -> HTTP {status}; retrying in {delay:.1f}s "
            f"(attempt {attempt} of {MAX_ATTEMPTS})",
            file=sys.stderr,
        )
        sleep(delay)
    raise AssertionError("unreachable: the loop either returns or raises")


# --- Credentials -----------------------------------------------------------


def load_api_key(env_file: Path | None = None) -> str:
    """Return `HEYGEN_API_KEY` from the environment, falling back to an env file.

    Test: `test_load_api_key_prefers_the_environment`
    """
    key = os.environ.get("HEYGEN_API_KEY", "").strip()
    if key:
        return key
    if env_file and env_file.is_file():
        match = re.search(
            r"^\s*(?:export\s+)?HEYGEN_API_KEY=(.+)$",
            env_file.read_text(),
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1).strip().strip("\"'")
    raise SystemExit(
        "No HEYGEN_API_KEY in the environment"
        + (f" or {env_file}" if env_file else "")
        + "."
    )


# --- Assets ----------------------------------------------------------------


def content_type_for(path: Path) -> str:
    """Map a file's suffix to the MIME type HeyGen accepts for it.

    Test: `test_content_type_covers_audio_and_image_suffixes`
    """
    suffix = path.suffix.lower()
    if suffix not in CONTENT_TYPES:
        raise SystemExit(f"{path}: unsupported asset type {suffix or '(none)'}.")
    return CONTENT_TYPES[suffix]


def file_sha256(path: Path) -> str:
    """Content hash used as the upload cache key."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    """Read the sha256-to-asset-id cache, treating a missing file as empty."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("assets", {})


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    """Write the cache back, sorted so a diff stays readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: cache[key] for key in sorted(cache)}
    path.write_text(json.dumps({"assets": ordered}, indent=2) + "\n")


def multipart_body(path: Path, content_type: str) -> tuple[bytes, str]:
    """Build a one-field `multipart/form-data` body for `POST /v3/assets`."""
    boundary = f"----breeze{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    body = head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def upload_asset(
    transport: Transport,
    api_key: str,
    path: Path,
    cache: dict[str, dict],
) -> str:
    """Upload one file, or return the id a previous upload of those bytes got.

    Test: `test_upload_reuses_a_cached_asset_id`,
    `test_upload_posts_multipart_and_caches`
    """
    if not path.is_file():
        raise SystemExit(f"Asset not found: {path}")
    size = path.stat().st_size
    if size > MAX_ASSET_BYTES:
        raise SystemExit(f"{path} is {size} bytes; HeyGen caps assets at 32 MB.")
    digest = file_sha256(path)
    cached = cache.get(digest)
    if cached:
        return cached["asset_id"]
    content_type = content_type_for(path)
    body, boundary_type = multipart_body(path, content_type)
    data = call(
        transport,
        "POST",
        "/assets",
        api_key,
        raw_body=body,
        extra_headers={"Content-Type": boundary_type, "Idempotency-Key": digest},
    )
    asset_id = data["asset_id"]
    cache[digest] = {
        "asset_id": asset_id,
        "name": path.name,
        "size_bytes": size,
        "mime_type": data.get("mime_type", content_type),
    }
    return asset_id


# --- Storyboard ------------------------------------------------------------


def scene_files(storyboard: dict, base: Path) -> list[Path]:
    """Every local file the storyboard's scenes reference, in first-use order.

    Test: `test_scene_files_lists_each_path_once`
    """
    paths: list[Path] = []
    for scene in storyboard["scenes"]:
        for key in ("image", "video", "audio"):
            value = scene.get(key)
            if not value:
                continue
            resolved = (
                (base / value).resolve()
                if not Path(value).is_absolute()
                else Path(value)
            )
            if resolved not in paths:
                paths.append(resolved)
    return paths


def _scene_request(scene: dict, assets: dict[Path, str], base: Path) -> dict:
    """Translate one storyboard scene into its v3 studio scene object."""

    def asset_of(key: str) -> str:
        value = scene[key]
        path = Path(value)
        return assets[path if path.is_absolute() else (base / value).resolve()]

    kind = scene.get("type")
    if kind == "avatar":
        avatar: dict[str, Any] = {
            "type": "avatar",
            "avatar_id": scene["avatar_id"],
            SCENE_AUDIO_KEY: asset_of("audio"),
        }
        engine = scene.get("engine")
        if engine:
            avatar["engine"] = {"type": engine}
        background = scene.get("background")
        if background:
            avatar["background"] = {"type": "color", "color": background}
        return {"type": "avatar_video", "input": avatar}
    if kind == "image":
        return {
            "type": "image",
            "source": {"type": "asset_id", "asset_id": asset_of("image")},
            SCENE_AUDIO_KEY: asset_of("audio"),
        }
    raise SystemExit(f"Unknown scene type {kind!r}; expected 'avatar' or 'image'.")


def build_request(
    storyboard: dict,
    assets: dict[Path, str],
    base: Path,
    *,
    burn_captions: bool = False,
) -> dict:
    """Build the `POST /v3/videos` body for a storyboard and its uploaded assets.

    Scene order is the storyboard's order; the globals (aspect ratio,
    resolution, title) come from its top level.

    Test: `test_build_request_orders_scenes_and_sets_globals`,
    `test_build_request_rejects_an_over_long_storyboard`
    """
    scenes = storyboard["scenes"]
    if not 1 <= len(scenes) <= 50:
        raise SystemExit(f"{len(scenes)} scenes; HeyGen studio allows 1 to 50.")
    request: dict[str, Any] = {
        "type": "studio",
        "aspect_ratio": storyboard.get("aspect_ratio", "16:9"),
        "resolution": storyboard.get("resolution", "1080p"),
        "scenes": [_scene_request(scene, assets, base) for scene in scenes],
    }
    if storyboard.get("title"):
        request["title"] = storyboard["title"]
    if burn_captions:
        request["caption"] = {"file_format": "srt", "style": "default"}
    return request


# --- Commands --------------------------------------------------------------


def command_upload(options: argparse.Namespace) -> int:
    """Upload one or more files and print each file's asset id."""
    api_key = load_api_key(options.env_file)
    cache = load_cache(options.cache)
    try:
        for path in options.files:
            asset_id = upload_asset(urllib_transport, api_key, Path(path), cache)
            print(f"{path}\t{asset_id}")
    finally:
        save_cache(options.cache, cache)
    return 0


def command_render(options: argparse.Namespace) -> int:
    """Upload whatever the storyboard needs, submit it, print the video id.

    Test: `test_render_uploads_then_submits`
    """
    storyboard = json.loads(options.storyboard.read_text())
    base = options.storyboard.resolve().parent
    api_key = load_api_key(options.env_file)
    cache = load_cache(options.cache)
    assets: dict[Path, str] = {}
    try:
        for path in scene_files(storyboard, base):
            assets[path] = upload_asset(urllib_transport, api_key, path, cache)
            print(f"asset {assets[path]}  {path.name}", file=sys.stderr)
    finally:
        save_cache(options.cache, cache)
    request = build_request(
        storyboard, assets, base, burn_captions=options.burn_captions
    )
    captioned = sum(1 for scene in storyboard["scenes"] if scene.get("caption"))
    if captioned and not options.burn_captions:
        print(
            f"{captioned} scenes carry caption text; v3 studio has no per-scene "
            "caption field, so it stays in the storyboard only.",
            file=sys.stderr,
        )
    if options.request_out:
        options.request_out.write_text(json.dumps(request, indent=2) + "\n")
    if options.dry_run:
        print(json.dumps(request, indent=2))
        return 0
    data = call(urllib_transport, "POST", "/videos", api_key, json_body=request)
    print(data["video_id"])
    return 0


def fetch_status(transport: Transport, api_key: str, video_id: str) -> dict:
    """One `GET /v3/videos/{id}` call."""
    return call(transport, "GET", f"/videos/{video_id}", api_key)


def poll(
    transport: Transport,
    api_key: str,
    video_id: str,
    *,
    timeout: float,
    interval: float,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Poll until the render finishes, fails, or the deadline passes.

    Test: `test_poll_stops_on_completion`, `test_poll_gives_up_at_the_deadline`
    """
    deadline = now() + timeout
    while True:
        data = fetch_status(transport, api_key, video_id)
        state = data.get("status", "unknown")
        print(
            f"{video_id} {state} ({now() - (deadline - timeout):.0f}s elapsed)",
            file=sys.stderr,
        )
        if state in TERMINAL_STATES:
            return data
        if now() >= deadline:
            raise SystemExit(f"{video_id} still {state} after {timeout:.0f}s.")
        sleep(min(interval, max(0.0, deadline - now())))


def command_status(options: argparse.Namespace) -> int:
    """Poll a render to completion and print the final status object."""
    api_key = load_api_key(options.env_file)
    data = poll(
        urllib_transport,
        api_key,
        options.video_id,
        timeout=options.timeout,
        interval=options.interval,
    )
    print(json.dumps(data, indent=2))
    return 0 if data.get("status") == "completed" else 1


def fetch_to_file(
    url: str, output: Path, opener: Callable = urllib.request.urlopen
) -> int:
    """Stream a presigned URL to disk, returning the byte count.

    Why: a 76 MB download that dies halfway used to leave a truncated MP4 at the
    final path, which every later check reads as "already downloaded". The bytes
    land in a sibling `.part` file and only become the target once the stream
    finishes, so an interruption leaves the target absent rather than wrong.

    Test: `test_fetch_to_file_leaves_nothing_behind_on_failure`,
    `test_fetch_to_file_replaces_the_target_on_success`
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "breeze-tts-heygen/1.0"}
    )
    partial = output.with_name(output.name + ".part")
    written = 0
    try:
        with opener(request, timeout=600) as response, partial.open("wb") as sink:
            for block in iter(lambda: response.read(1 << 20), b""):
                sink.write(block)
                written += len(block)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, output)
    return written


def command_download(options: argparse.Namespace) -> int:
    """Fetch a completed render's MP4, and its thumbnail when asked."""
    api_key = load_api_key(options.env_file)
    data = fetch_status(urllib_transport, api_key, options.video_id)
    if data.get("status") != "completed":
        raise SystemExit(f"{options.video_id} is {data.get('status')}, not completed.")
    written = fetch_to_file(data["video_url"], options.output)
    print(f"{options.output}\t{written} bytes")
    if options.thumbnail and data.get("thumbnail_url"):
        size = fetch_to_file(data["thumbnail_url"], options.thumbnail)
        print(f"{options.thumbnail}\t{size} bytes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Wire the four subcommands."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="JSON file mapping a file's sha256 to its HeyGen asset id.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser("upload", help="Upload files, print asset ids.")
    upload.add_argument("files", nargs="+")
    upload.set_defaults(handler=command_upload)

    render = subparsers.add_parser("render", help="Submit a storyboard as one video.")
    render.add_argument("storyboard", type=Path)
    render.add_argument("--burn-captions", action="store_true")
    render.add_argument("--dry-run", action="store_true")
    render.add_argument("--request-out", type=Path, help="Write the request body here.")
    render.set_defaults(handler=command_render)

    status = subparsers.add_parser("status", help="Poll a render to completion.")
    status.add_argument("video_id")
    status.add_argument("--timeout", type=float, default=1800.0)
    status.add_argument("--interval", type=float, default=15.0)
    status.set_defaults(handler=command_status)

    download = subparsers.add_parser("download", help="Fetch a finished MP4.")
    download.add_argument("video_id")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--thumbnail", type=Path)
    download.set_defaults(handler=command_download)
    return parser


def main() -> int:
    options = build_parser().parse_args()
    try:
        return options.handler(options)
    except HeyGenError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
