"""Frame extraction for Interlude.

This module is the *only* place in the codebase that shells out to ffmpeg.

ffmpeg (and its sibling ffprobe) are external binaries. They may simply be
absent on a grading machine, a CI runner, or a student's laptop. A missing
binary must degrade gracefully: a 400-video batch should skip visual
description and keep going, not crash on video #3. That is why absence is
signalled with a specific exception type (:class:`FFmpegUnavailable`) that
callers can catch and treat as "no visuals available", distinct from
:class:`FrameExtractionError`, which means ffmpeg was present but this
particular frame could not be produced.

Everything ffmpeg-shaped lives here so that the rest of the pipeline can be
tested with a plain in-memory callable matching the ``FrameLoader`` protocol.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Callable

__all__ = [
    "FFmpegUnavailable",
    "FrameExtractionError",
    "ffmpeg_available",
    "reset_ffmpeg_cache",
    "build_ffmpeg_command",
    "extract_frame",
    "make_frame_loader",
    "probe_duration",
]

log = logging.getLogger(__name__)

JPEG_MAGIC: bytes = b"\xff\xd8"

MIN_WIDTH: int = 128
MAX_WIDTH: int = 4096
MIN_QUALITY: int = 1
MAX_QUALITY: int = 31

# Cached result of shutil.which("ffmpeg"). None means "not yet looked up".
_FFMPEG_AVAILABLE: bool | None = None


class FFmpegUnavailable(RuntimeError):
    """Raised when the ffmpeg binary cannot be found on PATH."""


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg ran but did not produce a usable JPEG frame."""


def ffmpeg_available() -> bool:
    """Return True if an ``ffmpeg`` binary is on PATH.

    The answer is cached in a module-level variable: a 12,000-frame run
    should not stat every PATH entry 12,000 times. Tests that manipulate
    PATH must call :func:`reset_ffmpeg_cache` first.
    """
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        _FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
        log.debug("ffmpeg availability resolved to %s", _FFMPEG_AVAILABLE)
    return _FFMPEG_AVAILABLE


def reset_ffmpeg_cache() -> None:
    """Forget the cached ffmpeg lookup. Intended for tests."""
    global _FFMPEG_AVAILABLE
    _FFMPEG_AVAILABLE = None


def build_ffmpeg_command(
    source: str,
    timestamp: float,
    *,
    width: int = 1024,
    quality: int = 4,
) -> list[str]:
    """Build the ffmpeg argument list for a single-frame JPEG grab.

    Pure: validates its inputs and returns a list. Runs nothing, touches no
    filesystem. This is the part that is actually unit-testable.

    ``-ss`` is placed *before* ``-i`` on purpose. That is input seeking:
    ffmpeg jumps to the keyframe near the timestamp instead of decoding from
    frame zero. On a 50-minute lecture the difference is seconds versus
    minutes per frame.

    Raises:
        ValueError: on a negative timestamp, empty source, out-of-range
            width, or out-of-range quality.
    """
    if not source:
        raise ValueError("source must be a non-empty path or URL")
    if timestamp < 0:
        raise ValueError(f"timestamp must be >= 0, got {timestamp!r}")
    if not (MIN_WIDTH <= width <= MAX_WIDTH):
        raise ValueError(
            f"width must be between {MIN_WIDTH} and {MAX_WIDTH}, got {width!r}"
        )
    if not (MIN_QUALITY <= quality <= MAX_QUALITY):
        raise ValueError(
            f"quality must be between {MIN_QUALITY} and {MAX_QUALITY}, got {quality!r}"
        )

    return [
        "ffmpeg",
        "-ss",
        str(timestamp),
        "-i",
        source,
        "-frames:v",
        "1",
        # -2 preserves aspect ratio and rounds the height to an even number,
        # which JPEG/MJPEG encoders require.
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        str(quality),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-loglevel",
        "error",
        "-y",
        "-",
    ]


def extract_frame(
    source: str,
    timestamp: float,
    *,
    width: int = 1024,
    quality: int = 4,
    timeout: float = 30.0,
) -> bytes:
    """Extract one frame from ``source`` at ``timestamp`` as JPEG bytes.

    Raises:
        FFmpegUnavailable: ffmpeg is not installed.
        FrameExtractionError: ffmpeg failed, timed out, returned nothing, or
            returned something that is not a JPEG.
        ValueError: invalid arguments (propagated from
            :func:`build_ffmpeg_command`).
    """
    if not ffmpeg_available():
        raise FFmpegUnavailable(
            "ffmpeg was not found on PATH; install it (e.g. `brew install ffmpeg`) "
            "to enable visual description"
        )

    cmd = build_ffmpeg_command(source, timestamp, width=width, quality=quality)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FrameExtractionError(
            f"ffmpeg timed out after {timeout}s extracting t={timestamp}s from {source}"
        ) from exc

    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-200:]
        raise FrameExtractionError(
            f"ffmpeg exited {proc.returncode} extracting t={timestamp}s "
            f"from {source}: {tail}"
        )

    data = proc.stdout
    if not data:
        raise FrameExtractionError(
            f"ffmpeg produced no output for t={timestamp}s from {source}; "
            "the timestamp is likely beyond the end of the video"
        )

    if not data.startswith(JPEG_MAGIC):
        raise FrameExtractionError(
            f"output for t={timestamp}s from {source} is not a JPEG "
            f"(first bytes {data[:4]!r}); truncated or corrupt output would "
            "waste a vision-model call"
        )

    return data


def make_frame_loader(
    source: str,
    *,
    width: int = 1024,
    quality: int = 4,
    cache: bool = True,
) -> Callable[[float], bytes]:
    """Return a ``FrameLoader``: ``(timestamp: float) -> bytes``.

    If ``cache`` is True, results are memoized by timestamp rounded to two
    decimals. The description retry loop can ask for the same timestamp more
    than once, and re-shelling to ffmpeg for a frame already in hand is pure
    waste.

    The cache is **unbounded**. It is therefore a *per-video* cache, not a
    per-process one: a caller processing 12,000 videos must build one loader
    per video and let it go out of scope when that video is done. Holding a
    single loader across a whole batch will retain every JPEG ever decoded
    and exhaust memory.
    """
    memo: dict[float, bytes] = {}

    def load(timestamp: float) -> bytes:
        if not cache:
            return extract_frame(source, timestamp, width=width, quality=quality)
        key = round(timestamp, 2)
        hit = memo.get(key)
        if hit is not None:
            return hit
        data = extract_frame(source, timestamp, width=width, quality=quality)
        memo[key] = data
        return data

    return load


def probe_duration(source: str, *, timeout: float = 30.0) -> float | None:
    """Return the container duration of ``source`` in seconds, or None.

    Never raises. Returns None if ffprobe is missing, the call fails, or the
    output does not parse as a float.

    This exists because duration elsewhere in the pipeline is derived from
    the end time of the last spoken word, which understates the real duration
    whenever a lecture ends in silence (a slide left up, applause trimmed to
    quiet, a long pause before the recording stops). A real container
    duration lets the describer place a final visual cue correctly.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        log.debug("ffprobe not found; cannot probe duration of %s", source)
        return None
    if not source:
        return None

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.debug("ffprobe failed for %s: %s", source, exc)
        return None

    if proc.returncode != 0:
        log.debug("ffprobe exited %s for %s", proc.returncode, source)
        return None

    try:
        return float(proc.stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        log.debug("ffprobe output for %s did not parse as a float", source)
        return None


if __name__ == "__main__":
    # Runnable with no video file present and no ffmpeg installed.
    cmd = build_ffmpeg_command("v.mp4", 12.5)

    assert cmd[0] == "ffmpeg", cmd
    assert cmd.index("-ss") < cmd.index("-i"), "input seeking requires -ss before -i"
    assert "12.5" in cmd, cmd
    assert "v.mp4" in cmd, cmd
    assert cmd[-1] == "-", cmd
    assert "-frames:v" in cmd and cmd[cmd.index("-frames:v") + 1] == "1"
    scale = cmd[cmd.index("-vf") + 1]
    assert scale == "scale=1024:-2", scale
    assert ":-2" in scale, scale
    assert cmd[cmd.index("-q:v") + 1] == "4", cmd
    assert cmd[cmd.index("-f") + 1] == "image2pipe", cmd
    assert cmd[cmd.index("-vcodec") + 1] == "mjpeg", cmd
    assert cmd[cmd.index("-loglevel") + 1] == "error", cmd
    assert "-y" in cmd, cmd

    # Non-default width/quality land in the right slots.
    cmd2 = build_ffmpeg_command("lecture.mkv", 0, width=640, quality=2)
    assert cmd2[cmd2.index("-vf") + 1] == "scale=640:-2", cmd2
    assert cmd2[cmd2.index("-q:v") + 1] == "2", cmd2
    assert "0" in cmd2, cmd2

    # Validation.
    bad_args: list[tuple[str, float, int, int]] = [
        ("v.mp4", -0.1, 1024, 4),   # negative timestamp
        ("", 1.0, 1024, 4),         # empty source
        ("v.mp4", 1.0, 10, 4),      # width too small
        ("v.mp4", 1.0, 99999, 4),   # width too large
        ("v.mp4", 1.0, 1024, 0),    # quality too small
        ("v.mp4", 1.0, 1024, 99),   # quality too large
    ]
    for src, ts, w, q in bad_args:
        try:
            build_ffmpeg_command(src, ts, width=w, quality=q)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {(src, ts, w, q)!r}")

    # Boundaries are inclusive.
    build_ffmpeg_command("v.mp4", 0.0, width=MIN_WIDTH, quality=MIN_QUALITY)
    build_ffmpeg_command("v.mp4", 0.0, width=MAX_WIDTH, quality=MAX_QUALITY)

    # Loader factory: shape only, never invoked (that would need ffmpeg).
    loader = make_frame_loader("v.mp4")
    assert callable(loader)
    assert callable(make_frame_loader("v.mp4", cache=False))

    # Availability cache is a plain bool and survives a reset.
    assert isinstance(ffmpeg_available(), bool)
    reset_ffmpeg_cache()
    assert isinstance(ffmpeg_available(), bool)

    # probe_duration never raises, even on a path that cannot exist.
    assert probe_duration("/nonexistent/interlude/no-such-video.mp4") is None
    assert probe_duration("") is None

    print("frames.py self-check OK")
