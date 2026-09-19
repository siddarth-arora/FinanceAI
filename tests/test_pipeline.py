"""Verify the documented structured contract using a temporary Parquet dataset."""

import json
import socket
from unittest.mock import patch

import pytest

from src import pipeline
from src.config import TICKERS
from src.data_loader import load_stock_prices


def test_pipeline_json_contract_is_offline_and_preserves_dataset(monkeypatch, parquet_path):
    def forbid_network(*args, **kwargs):
        pytest.fail("The local pipeline attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(pipeline, "load_stock_prices", lambda: load_stock_prices(parquet_path))
    monkeypatch.setattr(pipeline, "RAW_PRICES_PATH", parquet_path)
    monkeypatch.setattr(pipeline, "FRONTIER_POINTS", 8)
    before = parquet_path.read_bytes()
    with patch.object(
        pipeline, "optimize_global_minimum_variance",
        wraps=pipeline.optimize_global_minimum_variance,
    ) as solve_gmv, patch(
        "src.optimizer.optimize_global_minimum_variance",
        side_effect=AssertionError("Frontier must reuse pipeline GMV"),
    ):
        payload = pipeline.run_mpt_analysis().to_dict()
    assert solve_gmv.call_count == 1
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert set(payload) == {"metadata", "annual_expected_returns", "annual_covariance",
                            "portfolios", "efficient_frontier"}
    metadata = payload["metadata"]
    assert metadata["assets"] == list(TICKERS)
    assert metadata["dataset_path"] == str(parquet_path)
    assert metadata["observations"] == 80
    assert metadata["trading_days"] == 252
    assert metadata["risk_free_rate"] == 0.04
    assert metadata["constraints"] == {
        "fully_invested": True, "long_only": True, "leverage_allowed": False,
    }
    assert len(payload["efficient_frontier"]) == 8
    for portfolio in [*payload["portfolios"].values(), *payload["efficient_frontier"]]:
        assert set(portfolio["weights"]) == set(TICKERS)
        assert sum(portfolio["weights"].values()) == pytest.approx(1.0, abs=1e-8)
        assert all(-1e-8 <= weight <= 1 + 1e-8 for weight in portfolio["weights"].values())
    assert parquet_path.read_bytes() == before
