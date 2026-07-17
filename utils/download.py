"""
Download utilities for the Fusion framework.
"""

import os
import shutil
import requests
from huggingface_hub import hf_hub_download

from tqdm import tqdm


def download_file(
    url: str,
    dest: str,
    show_progress: bool = True,
) -> None:
    """
    Download a file from a URL.

    Args:
        url: File URL.
        dest: Destination path.
        show_progress: Whether to display a progress bar.
    """
    directory = os.path.dirname(dest)

    if directory:
        os.makedirs(directory, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0))

    iterator = response.iter_content(chunk_size=8192)

    if show_progress:
        iterator = tqdm(
            iterator,
            total=total_size // 8192 + 1,
            unit="chunk",
            desc=os.path.basename(dest),
        )

    with open(dest, "wb") as file:
        for chunk in iterator:
            if chunk:
                file.write(chunk)


def download_from_hub(
    repo_id: str,
    filename: str,
    dest: str,
) -> None:
    """
    Download a file from the Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID.
        filename: File name inside the repository.
        dest: Destination path.
    """
    directory = os.path.dirname(dest)

    if directory:
        os.makedirs(directory, exist_ok=True)

    downloaded_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    shutil.copy(downloaded_file, dest)

