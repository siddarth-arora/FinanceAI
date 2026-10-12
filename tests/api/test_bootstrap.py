"""Dataset bootstrap: download, verify, validate, install atomically. Offline."""

import hashlib
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest

from api import bootstrap
from api.bootstrap import MAX_BYTES, DatasetSettings, ensure_dataset


URL = "https://github.com/owner/FinanceAI/releases/download/dataset-v1/stock_prices.parquet"


def settings(**values):
    # _env_file=None: tests must never read the developer's real .env.
    return DatasetSettings(_env_file=None, **values)


class Served:
    """A urlopen stand-in serving fixed bytes and recording the request."""

    def __init__(self, data: bytes):
        self.data, self.requests = data, []

    def __call__(self, request, timeout):
        self.requests.append(request)
        served = self

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, limit):
                return served.data[:limit]

        return Response()


@pytest.fixture
def artifact(dataset_path):
    return dataset_path.read_bytes()


@pytest.fixture
def serve(monkeypatch):
    def install(data):
        served = Served(data)
        monkeypatch.setattr(urllib.request, "urlopen", served)
        return served
    return install


def test_verified_artifact_is_downloaded_validated_and_installed(tmp_path, artifact, serve):
    target = tmp_path / "raw" / "stock_prices.parquet"
    serve(artifact)
    digest = hashlib.sha256(artifact).hexdigest()
    assert ensure_dataset(target, settings(url=URL, sha256=digest.upper())) == "downloaded"
    assert target.read_bytes() == artifact
    assert list(target.parent.iterdir()) == [target]   # no .part left behind


def test_checksum_mismatch_refuses_and_leaves_nothing(tmp_path, artifact, serve):
    target = tmp_path / "stock_prices.parquet"
    serve(artifact)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        ensure_dataset(target, settings(url=URL, sha256="0" * 64))
    assert list(tmp_path.iterdir()) == []


def test_correct_hash_of_a_corrupt_file_still_fails_validation(tmp_path, serve):
    """The hash proves identity, not validity; the loader still checks content."""
    garbage = b"not a parquet file"
    target = tmp_path / "stock_prices.parquet"
    serve(garbage)
    with pytest.raises(SystemExit, match="failed validation"):
        ensure_dataset(target, settings(url=URL, sha256=hashlib.sha256(garbage).hexdigest()))
    assert list(tmp_path.iterdir()) == []


def test_plain_http_is_rejected_before_any_request(tmp_path, serve):
    served = serve(b"")
    with pytest.raises(SystemExit, match="HTTPS"):
        ensure_dataset(tmp_path / "p.parquet", settings(url="http://example.com/p", sha256="0" * 64))
    assert served.requests == []


def test_oversized_artifact_is_rejected(tmp_path, serve):
    serve(b"x" * (MAX_BYTES + 10))
    with pytest.raises(SystemExit, match="size limit"):
        ensure_dataset(tmp_path / "p.parquet", settings(url=URL, sha256="0" * 64))
    assert list(tmp_path.iterdir()) == []


def test_matching_existing_file_is_not_downloaded(tmp_path, artifact, serve):
    target = tmp_path / "stock_prices.parquet"
    target.write_bytes(artifact)
    served = serve(b"")
    digest = hashlib.sha256(artifact).hexdigest()
    assert ensure_dataset(target, settings(url=URL, sha256=digest)) == "present"
    assert served.requests == []


def test_local_development_keeps_an_unpinned_existing_file(tmp_path, artifact, serve):
    target = tmp_path / "stock_prices.parquet"
    target.write_bytes(artifact)
    served = serve(b"")
    assert ensure_dataset(target, settings()) == "present"
    assert served.requests == []


def test_stale_existing_file_is_replaced_by_the_pinned_artifact(tmp_path, artifact, serve):
    target = tmp_path / "stock_prices.parquet"
    target.write_bytes(b"an older dataset")
    serve(artifact)
    digest = hashlib.sha256(artifact).hexdigest()
    assert ensure_dataset(target, settings(url=URL, sha256=digest)) == "downloaded"
    assert target.read_bytes() == artifact


def test_missing_file_without_configuration_exits(tmp_path):
    with pytest.raises(SystemExit, match="FINANCEAI_DATASET_URL"):
        ensure_dataset(tmp_path / "p.parquet", settings())


def test_token_is_sent_unredirected_and_never_reported(tmp_path, artifact, serve):
    served = serve(artifact)
    digest = hashlib.sha256(artifact).hexdigest()
    ensure_dataset(tmp_path / "p.parquet", settings(url=URL, sha256=digest, token="ghp_secret"))
    request = served.requests[0]
    assert request.unredirected_hdrs["Authorization"] == "Bearer ghp_secret"
    assert "Authorization" not in request.headers   # not forwarded on GitHub's redirect
    assert "ghp_secret" not in repr(settings(token="ghp_secret"))


def test_download_errors_name_neither_url_nor_token(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        Mock(side_effect=OSError(f"failed {URL} with ghp_secret")))
    with pytest.raises(SystemExit) as caught:
        ensure_dataset(tmp_path / "p.parquet",
                       settings(url=URL, sha256="0" * 64, token="ghp_secret"))
    message = str(caught.value)
    assert "OSError" in message
    assert URL not in message and "ghp_secret" not in message


def test_bootstrap_never_imports_market_data_code():
    source = Path(bootstrap.__file__).read_text(encoding="utf-8")
    assert "yfinance" not in source and "download_data" not in source
