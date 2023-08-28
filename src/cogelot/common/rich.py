from typing import cast

from rich import progress
from tqdm.rich import RateColumn


def create_progress_bar() -> progress.Progress:
    """Create a fully-featured progress bar with Rich."""
    progress_bar = progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(),
        progress.MofNCompleteColumn(),
        cast(progress.ProgressColumn, RateColumn(unit="it")),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
    )

    return progress_bar
