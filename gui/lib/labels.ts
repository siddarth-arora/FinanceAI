export type ProfileName = "low" | "medium" | "high";
export type BucketName = "conservative" | "balanced" | "growth" | "closest";

export const PROFILE_NAMES: ProfileName[] = ["low", "medium", "high"];

export const PROFILE_LABELS: Record<ProfileName, string> = {
  low: "Conservative",
  medium: "Balanced",
  high: "Growth",
};

export const BUCKET_LABELS: Record<string, string> = {
  conservative: "Conservative",
  balanced: "Balanced",
  growth: "Growth",
  closest: "Closest available",
};

export const SELECTION_RULES: Record<string, string> = {
  first_frontier_point: "The lowest-volatility point on the sampled efficient frontier (global minimum variance).",
  nearest_maximum_sharpe_volatility:
    "The sampled frontier point whose volatility is nearest the exact Maximum Sharpe portfolio.",
  nearest_interior_volatility_midpoint_at_or_above_medium:
    "An interior frontier point at or above Balanced, nearest the volatility midpoint between Maximum Sharpe and the frontier endpoint.",
};

export const WARNING_TEXT: Record<string, string> = {
  concentration_threshold_exceeded: "One holding exceeds 40% of the portfolio.",
  few_material_holdings: "Fewer than three holdings are above 1%; diversification is limited.",
  overlapping_profiles: "This portfolio is the same frontier point as another option.",
  frontier_endpoint_selected: "This is the frontier endpoint, typically a single-stock portfolio.",
  no_distinct_higher_interior_selection: "No distinct higher-risk interior point exists; Growth reuses Balanced.",
};

export function warningText(code: string): string {
  return WARNING_TEXT[code] ?? code.replaceAll("_", " ");
}

export const INFEASIBLE_TEXT: Record<string, string> = {
  return_floor_above_frontier: "No portfolio in this universe reached that return historically.",
  risk_ceiling_below_frontier: "Even the lowest-risk portfolio was more volatile than that limit.",
  risk_ceiling_and_return_floor_conflict: "No portfolio met both the risk limit and the return floor together.",
};

export const FALLBACK_TEXT: Record<string, string> = {
  missing_agent_dependencies: "AI client libraries are not installed on the server.",
  missing_groq_api_key: "The server has no Groq API key configured.",
  authentication_failed: "The AI provider rejected the server's credentials.",
  access_denied: "The AI provider denied access.",
  rate_limited: "The AI provider is rate limiting requests.",
  provider_unavailable: "The AI provider could not be reached.",
  request_rejected: "The AI provider rejected the request; check the configured model.",
  model_unavailable: "The configured AI model is not available; check GROQ_MODEL on the server.",
  incomplete_response: "The AI ran out of space before finishing its answer.",
  tool_call_limit: "The AI requested too many lookups.",
  invalid_tool_call: "The AI made an invalid lookup.",
  invalid_model_plan: "The AI returned an invalid answer plan.",
  explanation_failed: "The AI explanation failed.",
};

export const EMPHASIS_LABELS = {
  lower_risk: "Lower risk",
  balanced: "Balanced",
  higher_return: "Higher return",
} as const;
export type Emphasis = keyof typeof EMPHASIS_LABELS;

// Fixed per entity (validated all-pairs, light surface). Never assign by rank.
const GROUP_COLORS: Record<string, string> = {
  Technology: "var(--series-blue)",
  Healthcare: "var(--series-aqua)",
  Consumer: "var(--series-yellow)",
  Energy: "var(--series-violet)",
  Financials: "var(--series-red)",
  Other: "var(--series-other)",
};

export function groupColor(group: string): string {
  return GROUP_COLORS[group] ?? "var(--series-other)";
}
