from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from api.schemas.common import Breakdown, Currency, Freq, Metadata, ProfileName, RangeKey, Weights
from api.schemas.profiles import BucketsResult, Profile


class HealthResponse(BaseModel):
    status: Literal["ok"]
    dataset_loaded: bool
    llm_configured: bool
    plans_enabled: bool


class MetadataResponse(Metadata):
    disclaimer: str
    currency_notice: str


class AssetStat(BaseModel):
    ticker: str
    name: str
    sector: str
    group: str
    annual_expected_return: float
    annual_volatility: float


class UniverseResponse(BaseModel):
    assets: list[AssetStat]
    metadata: Metadata
    disclaimer: str


class CorrelationResponse(BaseModel):
    assets: list[str]
    matrix: list[list[float]]


class PriceHistoryResponse(BaseModel):
    range: RangeKey
    freq: Freq
    start_date: str
    end_date: str
    assets: list[str]
    series: list[dict[str, float | str]]


class FrontierPoint(BaseModel):
    index: int
    target_return: float | None
    expected_return: float
    variance: float
    volatility: float
    sharpe_ratio: float
    weights: Weights


class OptimizedPortfolio(BaseModel):
    expected_return: float
    variance: float
    volatility: float
    sharpe_ratio: float
    weights: Weights


class FrontierAsset(BaseModel):
    ticker: str
    name: str
    group: str
    expected_return: float
    volatility: float


class FrontierResponse(BaseModel):
    points: list[FrontierPoint]
    gmv: OptimizedPortfolio
    max_sharpe: OptimizedPortfolio
    profiles: dict[ProfileName, int]
    assets: list[FrontierAsset]
    risk_free_rate: float


class PortfolioRef(BaseModel):
    kind: Literal["profile", "frontier_point"]
    profile: ProfileName | None
    frontier_index: int


class SeriesPoint(BaseModel):
    date: str
    value: str
    benchmark: str | None


class PerformanceSummary(BaseModel):
    start_value: str
    end_value: str
    gain: str
    total_return: float
    cagr: float | None
    realized_volatility: float | None
    max_drawdown: float


class PerformanceResponse(BaseModel):
    portfolio: PortfolioRef
    method: str
    amount: str
    currency: Currency
    range: RangeKey
    freq: Freq
    start_date: str
    end_date: str
    series: list[SeriesPoint]
    summary: PerformanceSummary
    benchmark: Literal["equal_weight"] | None
    benchmark_summary: PerformanceSummary | None
    notice: str
    currency_notice: str


class AllocationDisplay(BaseModel):
    amounts: dict[str, str]
    hidden_assets: list[str]
    hidden_amount: str
    total: str
    dust_threshold: float
    rounding_adjustments: dict[str, str]
    method: str
    notice: str


class AllocationResult(BaseModel):
    mode: Literal["fractional"]
    amount: str
    currency: Currency
    target_profile: Profile
    target_amounts: dict[str, str]
    numerical_residual: str
    display: AllocationDisplay
    notice: str


class AllocationResponse(AllocationResult):
    breakdown: Breakdown
    currency_notice: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class Explanation(BaseModel):
    text: str
    #: The model's own reply, present only when it passed number validation.
    #: Every figure in it appears verbatim in a cited fact; `text` remains the
    #: Python-rendered grounding for it.
    answer: str | None = None
    fact_ids: list[str]
    metadata: Metadata
    disclaimer: str
    status: Literal["supported", "unsupported"]
    mode: Literal["offline", "llm", "fallback"]
    model: str | None
    fallback_reason: str | None
    #: Set when prose was discarded for stating an unverifiable figure.
    answer_rejected: str | None = None
    tool_calls: list[ToolCall]
    provider: str | None
    question_answered: bool


class NewsItemOut(BaseModel):
    ticker: str | None
    headline: str
    source: str
    published: str
    url: str


class ExplainResponse(BaseModel):
    profiles: dict[ProfileName, Profile]
    buckets: BucketsResult | None
    allocation: AllocationResult | None
    news: list[NewsItemOut]
    news_status: str
    explanation: Explanation
    orchestration: Literal["langgraph", "sequential"]


class PlanSummary(BaseModel):
    id: str
    name: str
    note: str | None
    created_at: datetime
    amount: str
    currency: Currency
    profile: ProfileName
    hide_dust: bool
    dataset_start: str
    dataset_end: str
    dataset_matches_current: bool


class PlanDetail(PlanSummary):
    allocation: AllocationResponse


class PlanList(BaseModel):
    items: list[PlanSummary]
    total: int
    limit: int
    offset: int
