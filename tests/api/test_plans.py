from decimal import Decimal

import pytest


BODY = {"name": "  Retirement  ", "note": "First try", "amount": "250000.5", "currency": "INR",
        "profile": "high", "hide_dust": True}


def test_create_get_list_delete(client):
    created = client.post("/api/plans", json=BODY)
    assert created.status_code == 201
    plan = created.json()
    assert plan["name"] == "Retirement" and plan["amount"] == "250000.50"
    assert plan["dataset_matches_current"] is True and plan["created_at"].endswith(("Z", "+00:00"))
    allocation = plan["allocation"]
    assert allocation["display"]["total"] == "250000.50" and allocation["target_profile"]["name"] == "high"
    assert "dataset_path" not in created.text

    fetched = client.get(f"/api/plans/{plan['id']}").json()
    assert fetched["allocation"] == allocation

    client.post("/api/plans", json={**BODY, "name": "Second", "profile": "low"})
    listing = client.get("/api/plans?limit=1").json()
    assert listing["total"] == 2 and listing["limit"] == 1 and len(listing["items"]) == 1
    assert "allocation" not in listing["items"][0]

    assert client.delete(f"/api/plans/{plan['id']}").status_code == 204
    assert client.get(f"/api/plans/{plan['id']}").status_code == 404
    assert client.delete(f"/api/plans/{plan['id']}").json()["error"]["code"] == "not_found"
    assert client.get("/api/plans").json()["total"] == 1


def test_server_recomputes_amounts(client):
    rejected = client.post("/api/plans", json={**BODY, "target_amounts": {"AAPL": "999"}})
    assert rejected.status_code == 422
    plan = client.post("/api/plans", json=BODY).json()
    amounts = plan["allocation"]["display"]["amounts"]
    total = sum(Decimal(value) for value in amounts.values()) + Decimal(plan["allocation"]["display"]["hidden_amount"])
    assert total == Decimal("250000.50")


@pytest.mark.parametrize("body", [{**BODY, "name": "   "}, {**BODY, "name": "x" * 81},
                                  {**BODY, "note": "n" * 501}, {**BODY, "amount": "1.234"}])
def test_plan_validation(client, body):
    assert client.post("/api/plans", json=body).status_code == 422


def test_plan_csv_exports_every_ticker_even_with_hidden_dust(client):
    plan = client.post("/api/plans", json=BODY).json()
    response = client.get(f"/api/plans/{plan['id']}/export.csv")
    assert response.status_code == 200
    assert 'filename="financeai-plan-high-inr.csv"' in response.headers["content-disposition"]
    rows = [line for line in response.text.splitlines()[1:] if not line.startswith("#")]
    assert len(rows) == 10


def test_plans_persist_across_restarts(make_client, tmp_path):
    url = f"sqlite:///{(tmp_path / 'plans.db').as_posix()}"
    first = make_client(database_url=url)
    plan_id = first.post("/api/plans", json=BODY).json()["id"]
    first.__exit__(None, None, None)
    second = make_client(database_url=url)
    assert second.get(f"/api/plans/{plan_id}").json()["name"] == "Retirement"


def test_plans_can_be_disabled(make_client):
    client = make_client(enable_plans=False)
    assert client.get("/api/health").json()["plans_enabled"] is False
    assert client.get("/api/plans").status_code == 404
    assert client.post("/api/plans", json=BODY).status_code == 404


@pytest.mark.parametrize("given,expected", [
    ("postgresql://u:p@ep-x.neon.tech/db?sslmode=require",
     "postgresql+psycopg://u:p@ep-x.neon.tech/db?sslmode=require"),
    ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
    ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
    ("sqlite:///data/app/financeai.db", "sqlite:///data/app/financeai.db"),
])
def test_pasted_postgres_urls_use_the_installed_driver(given, expected):
    """Neon's URL pasted as-is must not select psycopg2, which is not installed."""
    from api.db import normalize_database_url
    assert normalize_database_url(given) == expected
