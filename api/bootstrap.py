"""Fetch the frozen dataset artifact before the API starts. Not a market-data download.

Run: python -m api.bootstrap (from FinanceAI/). It never contacts Yahoo; it
downloads our own versioned release asset, checks its SHA-256, validates it with
the baseline loader and only then installs it. Any failure exits non-zero, so
a server can never start on a missing or unverified dataset.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import PROJECT_ROOT, RAW_PRICES_PATH
from src.data_loader import load_stock_prices


MAX_BYTES = 20 * 1024 * 1024        # the real file is ~105 KB; refuse anything absurd
TIMEOUT_SECONDS = 60


class DatasetSettings(BaseSettings):
    """FINANCEAI_DATASET_* from the environment (Render) or FinanceAI/.env (local)."""

    model_config = SettingsConfigDict(env_prefix="FINANCEAI_DATASET_",
                                      env_file=PROJECT_ROOT / ".env", extra="ignore")

    url: str | None = None
    sha256: str | None = None
    token: SecretStr | None = None   # only for a private repository's release asset


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _request(url: str, token: SecretStr | None) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"User-Agent": "financeai-bootstrap"})
    if token is not None and token.get_secret_value():
        request.add_header("Accept", "application/octet-stream")
        # Unredirected: never forward the token to the storage host GitHub redirects to.
        request.add_unredirected_header("Authorization", f"Bearer {token.get_secret_value()}")
    return request


def ensure_dataset(path: Path = RAW_PRICES_PATH, settings: DatasetSettings | None = None) -> str:
    """Return "present" or "downloaded"; raise SystemExit on any failure.

    An existing file is kept when no checksum is configured (local development)
    or when it matches the configured one; otherwise the pinned artifact is
    fetched. Error messages never include the URL or token.
    """
    settings = settings or DatasetSettings()
    expected = (settings.sha256 or "").strip().lower()
    if path.is_file() and (not expected or _sha256(path.read_bytes()) == expected):
        load_stock_prices(path)
        return "present"
    if not settings.url or not expected:
        raise SystemExit("Dataset missing: set FINANCEAI_DATASET_URL and FINANCEAI_DATASET_SHA256.")
    if not settings.url.startswith("https://"):
        raise SystemExit("FINANCEAI_DATASET_URL must use HTTPS.")
    try:
        with urllib.request.urlopen(_request(settings.url, settings.token),
                                    timeout=TIMEOUT_SECONDS) as response:
            data = response.read(MAX_BYTES + 1)
    except Exception as error:
        raise SystemExit(f"Dataset download failed ({type(error).__name__}).") from None
    if len(data) > MAX_BYTES:
        raise SystemExit("Dataset artifact exceeds the size limit.")
    if _sha256(data) != expected:
        raise SystemExit("Dataset checksum mismatch; refusing to start.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".parquet.part")
    try:
        partial.write_bytes(data)
        load_stock_prices(partial)    # schema, ticker order, NaN, positivity: before install
        os.replace(partial, path)     # atomic: never a half-written or unvalidated Parquet
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"Dataset artifact failed validation ({type(error).__name__}).") from None
    return "downloaded"


if __name__ == "__main__":
    print(f"dataset: {ensure_dataset()}", file=sys.stderr)
