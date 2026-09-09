"""Tests for the HeyGen v3 client's offline halves.

Credential loading, content-type detection, the upload cache, request
construction and the polling loop are all pure or filesystem-local. Every test
that would otherwise reach api.heygen.com drives a stub transport instead, so
the suite never uploads an asset or spends a credit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "heygen_video.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("heygen_video", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["heygen_video"] = module
    spec.loader.exec_module(module)
    return module


heygen_video = _load_module()


class StubTransport:
    """A transport that replays canned responses and records what it was sent."""

    def __init__(self, responses: list[tuple[int, dict, bytes]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict, bytes | None]] = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, dict(headers), body))
        return self.responses.pop(0)


def ok(payload: dict) -> tuple[int, dict, bytes]:
    """A 200 carrying HeyGen's usual `data` wrapper."""
    return 200, {}, json.dumps({"data": payload}).encode()


def error(status: int, message: str, headers: dict | None = None):
    """A non-2xx carrying HeyGen's usual `error` object."""
    body = json.dumps({"error": {"code": "e", "message": message}}).encode()
    return status, headers or {}, body


# --- Credentials and content types -----------------------------------------


def test_load_api_key_prefers_the_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text("HEYGEN_API_KEY=from-file\n")
    monkeypatch.setenv("HEYGEN_API_KEY", "from-env")
    assert heygen_video.load_api_key(env_file) == "from-env"


def test_load_api_key_falls_back_to_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text('OTHER=1\nexport HEYGEN_API_KEY="from-file"\n')
    monkeypatch.delenv("HEYGEN_API_KEY", raising=False)
    assert heygen_video.load_api_key(env_file) == "from-file"


def test_load_api_key_without_a_key_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("HEYGEN_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        heygen_video.load_api_key(tmp_path / "absent")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.wav", "audio/wav"),
        ("a.mp3", "audio/mpeg"),
        ("a.png", "image/png"),
        ("a.jpeg", "image/jpeg"),
        ("a.mp4", "video/mp4"),
    ],
)
def test_content_type_covers_audio_and_image_suffixes(name, expected):
    assert heygen_video.content_type_for(Path(name)) == expected


def test_content_type_rejects_an_unsupported_suffix():
    with pytest.raises(SystemExit):
        heygen_video.content_type_for(Path("notes.txt"))


# --- Transport and errors --------------------------------------------------


def test_call_raises_with_the_response_body():
    transport = StubTransport([error(400, "avatar_id required")])
    with pytest.raises(heygen_video.HeyGenError) as caught:
        heygen_video.call(transport, "POST", "/videos", "key", json_body={})
    assert "avatar_id required" in str(caught.value)
    assert caught.value.status == 400


def test_call_does_not_retry_a_plain_client_error():
    transport = StubTransport([error(400, "avatar_id required"), ok({"video_id": "x"})])
    with pytest.raises(heygen_video.HeyGenError):
        heygen_video.call(transport, "POST", "/videos", "key", json_body={})
    assert len(transport.calls) == 1


def test_call_retries_a_429_then_succeeds():
    slept: list[float] = []
    transport = StubTransport(
        [
            error(429, "slow down", {"Retry-After": "3"}),
            error(503, "upstream busy"),
            ok({"video_id": "vid_9"}),
        ]
    )
    data = heygen_video.call(
        transport, "POST", "/videos", "key", json_body={}, sleep=slept.append
    )
    assert data == {"video_id": "vid_9"}
    assert len(transport.calls) == 3
    # The 429's Retry-After wins; the 503 has none, so attempt 2 backs off 4 s.
    assert slept == [3.0, 4.0]


def test_call_gives_up_after_max_attempts():
    slept: list[float] = []
    transport = StubTransport([error(500, "boom")] * heygen_video.MAX_ATTEMPTS)
    with pytest.raises(heygen_video.HeyGenError) as caught:
        heygen_video.call(transport, "GET", "/videos/v", "key", sleep=slept.append)
    assert caught.value.status == 500
    assert len(transport.calls) == heygen_video.MAX_ATTEMPTS
    assert len(slept) == heygen_video.MAX_ATTEMPTS - 1


def test_retry_delay_honours_retry_after():
    assert heygen_video.retry_delay({"Retry-After": "7"}, 1) == 7.0
    assert heygen_video.retry_delay({"retry-after": "7"}, 3) == 7.0
    capped = heygen_video.retry_delay({"Retry-After": "99999"}, 1)
    assert capped == heygen_video.MAX_RETRY_AFTER_S


def test_retry_delay_backs_off_exponentially():
    assert [heygen_video.retry_delay({}, attempt) for attempt in (1, 2, 3)] == [
        2.0,
        4.0,
        8.0,
    ]
    # An HTTP-date Retry-After is not a number, so the backoff still applies.
    assert (
        heygen_video.retry_delay({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 1)
        == 2.0
    )


def test_call_unwraps_the_data_object():
    transport = StubTransport([ok({"video_id": "vid_1"})])
    assert heygen_video.call(transport, "GET", "/videos/vid_1", "key") == {
        "video_id": "vid_1"
    }
    method, url, headers, _ = transport.calls[0]
    assert (method, url) == ("GET", "https://api.heygen.com/v3/videos/vid_1")
    assert headers["X-Api-Key"] == "key"


# --- Upload cache ----------------------------------------------------------


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_upload_reuses_a_cached_asset_id(tmp_path):
    source = write_bytes(tmp_path / "intro.wav", b"RIFFfake")
    cache = {heygen_video.file_sha256(source): {"asset_id": "asset_cached"}}
    transport = StubTransport([])
    assert heygen_video.upload_asset(transport, "key", source, cache) == "asset_cached"
    assert transport.calls == []


def test_upload_posts_multipart_and_caches(tmp_path):
    source = write_bytes(tmp_path / "hero.png", b"\x89PNG-body")
    cache: dict[str, dict] = {}
    transport = StubTransport([ok({"asset_id": "asset_new", "mime_type": "image/png"})])
    assert heygen_video.upload_asset(transport, "key", source, cache) == "asset_new"
    method, url, headers, body = transport.calls[0]
    assert (method, url) == ("POST", "https://api.heygen.com/v3/assets")
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert headers["Idempotency-Key"] == heygen_video.file_sha256(source)
    assert b'name="file"; filename="hero.png"' in body
    assert b"\x89PNG-body" in body
    assert cache[heygen_video.file_sha256(source)]["asset_id"] == "asset_new"


def test_upload_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "cache.json"
    heygen_video.save_cache(path, {"sha": {"asset_id": "a1", "name": "x.wav"}})
    assert heygen_video.load_cache(path)["sha"]["asset_id"] == "a1"
    assert heygen_video.load_cache(tmp_path / "absent.json") == {}


def test_upload_refuses_an_oversized_file(tmp_path, monkeypatch):
    source = write_bytes(tmp_path / "big.wav", b"x" * 64)
    monkeypatch.setattr(heygen_video, "MAX_ASSET_BYTES", 8)
    with pytest.raises(SystemExit):
        heygen_video.upload_asset(StubTransport([]), "key", source, {})


# --- Request construction --------------------------------------------------


def sample_storyboard() -> dict:
    return {
        "title": "Delegation test",
        "aspect_ratio": "16:9",
        "resolution": "1080p",
        "scenes": [
            {
                "type": "avatar",
                "avatar_id": "look_1",
                "engine": "avatar_iii",
                "audio": "audio/intro.wav",
            },
            {
                "type": "image",
                "image": "images/hero.png",
                "audio": "audio/body/chunk_0.wav",
                "caption": "first chunk",
            },
            {
                "type": "avatar",
                "avatar_id": "look_1",
                "engine": "avatar_iii",
                "audio": "audio/closing.wav",
            },
        ],
    }


def resolved_assets(base: Path) -> dict[Path, str]:
    return {
        (base / "audio/intro.wav").resolve(): "asset_intro",
        (base / "images/hero.png").resolve(): "asset_hero",
        (base / "audio/body/chunk_0.wav").resolve(): "asset_body_0",
        (base / "audio/closing.wav").resolve(): "asset_closing",
    }


def test_scene_files_lists_each_path_once(tmp_path):
    storyboard = sample_storyboard()
    storyboard["scenes"][2]["audio"] = "audio/intro.wav"
    paths = heygen_video.scene_files(storyboard, tmp_path)
    assert [path.name for path in paths] == ["intro.wav", "hero.png", "chunk_0.wav"]


def test_build_request_orders_scenes_and_sets_globals(tmp_path):
    request = heygen_video.build_request(
        sample_storyboard(), resolved_assets(tmp_path), tmp_path
    )
    assert request["type"] == "studio"
    assert request["aspect_ratio"] == "16:9"
    assert request["resolution"] == "1080p"
    assert request["title"] == "Delegation test"
    assert [scene["type"] for scene in request["scenes"]] == [
        "avatar_video",
        "image",
        "avatar_video",
    ]
    first = request["scenes"][0]["input"]
    assert first == {
        "type": "avatar",
        "avatar_id": "look_1",
        "audio_asset_id": "asset_intro",
        "engine": {"type": "avatar_iii"},
    }
    middle = request["scenes"][1]
    assert middle["source"] == {"type": "asset_id", "asset_id": "asset_hero"}
    assert middle["audio_asset_id"] == "asset_body_0"
    assert "caption" not in middle
    assert request["scenes"][2]["input"]["audio_asset_id"] == "asset_closing"


def test_build_request_burns_captions_only_when_asked(tmp_path):
    plain = heygen_video.build_request(
        sample_storyboard(), resolved_assets(tmp_path), tmp_path
    )
    assert "caption" not in plain
    burned = heygen_video.build_request(
        sample_storyboard(), resolved_assets(tmp_path), tmp_path, burn_captions=True
    )
    assert burned["caption"] == {"file_format": "srt", "style": "default"}


def test_build_request_rejects_an_over_long_storyboard(tmp_path):
    scene = sample_storyboard()["scenes"][1]
    storyboard = {"scenes": [dict(scene) for _ in range(51)]}
    with pytest.raises(SystemExit):
        heygen_video.build_request(storyboard, resolved_assets(tmp_path), tmp_path)


def test_build_request_rejects_an_unknown_scene_type(tmp_path):
    storyboard = {"scenes": [{"type": "gif", "audio": "audio/intro.wav"}]}
    with pytest.raises(SystemExit):
        heygen_video.build_request(storyboard, resolved_assets(tmp_path), tmp_path)


# --- Render, poll ----------------------------------------------------------


def test_render_uploads_then_submits(tmp_path, monkeypatch):
    base = tmp_path
    for relative in (
        "audio/intro.wav",
        "images/hero.png",
        "audio/body/chunk_0.wav",
        "audio/closing.wav",
    ):
        write_bytes(base / relative, relative.encode())
    storyboard_path = base / "storyboard.json"
    storyboard_path.write_text(json.dumps(sample_storyboard()))
    transport = StubTransport(
        [
            ok({"asset_id": "asset_intro"}),
            ok({"asset_id": "asset_hero"}),
            ok({"asset_id": "asset_body_0"}),
            ok({"asset_id": "asset_closing"}),
            ok({"video_id": "vid_42", "status": "pending"}),
        ]
    )
    monkeypatch.setattr(heygen_video, "urllib_transport", transport)
    monkeypatch.setenv("HEYGEN_API_KEY", "key")
    options = heygen_video.build_parser().parse_args(
        [
            "--cache",
            str(base / "cache.json"),
            "render",
            str(storyboard_path),
            "--request-out",
            str(base / "request.json"),
        ]
    )
    assert options.handler(options) == 0
    assert [call[1] for call in transport.calls[:4]] == [
        "https://api.heygen.com/v3/assets"
    ] * 4
    submitted = json.loads(transport.calls[4][3])
    assert submitted["scenes"][1]["source"]["asset_id"] == "asset_hero"
    assert json.loads((base / "request.json").read_text()) == submitted
    cached = heygen_video.load_cache(base / "cache.json")
    assert sorted(entry["asset_id"] for entry in cached.values()) == [
        "asset_body_0",
        "asset_closing",
        "asset_hero",
        "asset_intro",
    ]


def test_render_dry_run_submits_nothing(tmp_path, monkeypatch, capsys):
    base = tmp_path
    for relative in (
        "audio/intro.wav",
        "images/hero.png",
        "audio/body/chunk_0.wav",
        "audio/closing.wav",
    ):
        write_bytes(base / relative, relative.encode())
    storyboard_path = base / "storyboard.json"
    storyboard_path.write_text(json.dumps(sample_storyboard()))
    cache = {
        heygen_video.file_sha256(base / relative): {"asset_id": f"asset_{index}"}
        for index, relative in enumerate(
            (
                "audio/intro.wav",
                "images/hero.png",
                "audio/body/chunk_0.wav",
                "audio/closing.wav",
            )
        )
    }
    heygen_video.save_cache(base / "cache.json", cache)
    transport = StubTransport([])
    monkeypatch.setattr(heygen_video, "urllib_transport", transport)
    monkeypatch.setenv("HEYGEN_API_KEY", "key")
    options = heygen_video.build_parser().parse_args(
        [
            "--cache",
            str(base / "cache.json"),
            "render",
            str(storyboard_path),
            "--dry-run",
        ]
    )
    assert options.handler(options) == 0
    assert transport.calls == []
    assert json.loads(capsys.readouterr().out)["type"] == "studio"


class StubStream:
    """A response body that yields `blocks`, then optionally raises mid-stream."""

    def __init__(self, blocks: list[bytes], fail_after: int | None = None) -> None:
        self.blocks = list(blocks)
        self.fail_after = fail_after
        self.served = 0

    def read(self, _size):
        if self.fail_after is not None and self.served == self.fail_after:
            raise ConnectionResetError("connection reset mid-stream")
        if not self.blocks:
            return b""
        self.served += 1
        return self.blocks.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_fetch_to_file_replaces_the_target_on_success(tmp_path):
    target = tmp_path / "nested" / "video.mp4"
    stream = StubStream([b"abc", b"def"])
    written = heygen_video.fetch_to_file(
        "https://example/x.mp4", target, opener=lambda *_a, **_k: stream
    )
    assert written == 6
    assert target.read_bytes() == b"abcdef"
    assert not target.with_name("video.mp4.part").exists()


def test_fetch_to_file_leaves_nothing_behind_on_failure(tmp_path):
    target = tmp_path / "video.mp4"
    stream = StubStream([b"abc", b"def"], fail_after=1)
    with pytest.raises(ConnectionResetError):
        heygen_video.fetch_to_file(
            "https://example/x.mp4", target, opener=lambda *_a, **_k: stream
        )
    assert not target.exists()
    assert not target.with_name("video.mp4.part").exists()
    assert list(tmp_path.iterdir()) == []


class FakeClock:
    """A monotonic clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_poll_stops_on_completion():
    clock = FakeClock()
    transport = StubTransport(
        [
            ok({"status": "processing"}),
            ok({"status": "completed", "video_url": "https://example/x.mp4"}),
        ]
    )
    data = heygen_video.poll(
        transport,
        "key",
        "vid_1",
        timeout=100,
        interval=10,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert data["status"] == "completed"
    assert len(transport.calls) == 2


def test_poll_gives_up_at_the_deadline():
    clock = FakeClock()
    transport = StubTransport([ok({"status": "processing"})] * 5)
    with pytest.raises(SystemExit):
        heygen_video.poll(
            transport,
            "key",
            "vid_1",
            timeout=20,
            interval=10,
            now=clock.now,
            sleep=clock.sleep,
        )


def test_poll_returns_a_failure_with_its_message():
    clock = FakeClock()
    transport = StubTransport(
        [ok({"status": "failed", "failure_message": "audio too long"})]
    )
    data = heygen_video.poll(
        transport,
        "key",
        "vid_1",
        timeout=30,
        interval=5,
        now=clock.now,
        sleep=clock.sleep,
    )
    assert data["failure_message"] == "audio too long"
