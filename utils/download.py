"""Asset and model download helpers.

Provides ``download_file`` for generic HTTP downloads with a progress
bar, and ``download_from_hub`` as a thin wrapper around the
Hugging Face Hub client.

Expected config params: None — all functions accept explicit arguments.
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from tqdm import tqdm


def download_file(
    url: str,
    dest: str,
    show_progress: bool = True,
    chunk_size: int = 8192,
) -> str:
    """Stream-download a file from *url* to *dest*.

    Args:
        url (str): Source URL.
        dest (str): Local destination path (file or directory).
            If *dest* is a directory, the filename is inferred from
            the URL.
        show_progress (bool): Show a ``tqdm`` progress bar.
        chunk_size (int): Bytes per read chunk.

    Returns:
        str: Absolute path to the downloaded file.

    Raises:
        requests.HTTPError: On non-2xx responses.
        OSError: On filesystem write failures.
    """
    if os.path.isdir(dest):
        filename = url.split("/")[-1].split("?")[0] or "download"
        dest = os.path.join(dest, filename)

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    progress = tqdm(
        total=total or None,
        unit="B",
        unit_scale=True,
        desc=os.path.basename(dest),
        disable=not show_progress,
    )

    with open(dest, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            progress.update(len(chunk))

    progress.close()
    return os.path.abspath(dest)


def download_from_hub(
    repo_id: str,
    filename: str,
    dest: str,
    revision: Optional[str] = None,
) -> str:
    """Download a single file from a Hugging Face Hub repository.

    Thin wrapper around ``huggingface_hub.hf_hub_download`` that
    copies the cached file to *dest*.

    Args:
        repo_id (str): Hub repository identifier
            (e.g. ``"facebook/bart-large"``).
        filename (str): Path inside the repository
            (e.g. ``"config.json"``).
        dest (str): Local destination path.
        revision (Optional[str]): Git revision (branch, tag, or SHA).

    Returns:
        str: Absolute path to the downloaded file.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for download_from_hub(). "
            "Install it with: pip install huggingface_hub"
        ) from exc

    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)

    import shutil
    shutil.copy2(cached_path, dest)

    return os.path.abspath(dest)
