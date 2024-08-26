import subprocess
from contextlib import suppress


def verify_ffmpeg_is_available() -> None:
    """Verify ffmpeg is available."""
    is_available = False
    with suppress(FileNotFoundError):
        outcome = subprocess.run(  # noqa: S603
            ["ffmpeg", "-version"],  # noqa: S607
            capture_output=True,
            check=False,
        )
        is_available = outcome.returncode == 0

    if not is_available:
        raise RuntimeError("ffmpeg is not installed. Please install ffmpeg to run the evaluation.")
