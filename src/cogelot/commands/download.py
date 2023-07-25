import os

import datasets
import typer


app = typer.Typer()


@app.command()
def download_from_hf(
    repo_id: str = "amitkparekh/vima", num_workers: int | None = os.cpu_count()
) -> None:
    """Download the dataset from HF."""
    datasets.load_dataset(repo_id, num_proc=num_workers)


if __name__ == "__main__":
    app()
