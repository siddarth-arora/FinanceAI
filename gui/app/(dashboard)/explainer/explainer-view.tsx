"use client";

import clsx from "clsx";
import { Bot, Loader2, Send, Sparkles, Square } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { useRef, useState } from "react";
import { PageSkeleton } from "@/components/common/blocks";
import { PageHeader } from "@/components/shell/page-header";
import { Badge, Button, Card, CardHeader, EmptyState, ErrorCard, Footnote, Note, Select, Switch } from "@/components/ui/primitives";
import { ApiError, type Schemas } from "@/lib/api/client";
import { useExplain, useHealth, type ExplainParams } from "@/lib/api/queries";
import { isValidAmount, money, type Currency } from "@/lib/format";
import { EMPHASIS_LABELS, FALLBACK_TEXT, PROFILE_LABELS, PROFILE_NAMES, type Emphasis, type ProfileName } from "@/lib/labels";
import { useSettings, useSettingsHydrated } from "@/lib/store/settings";

const SUGGESTIONS = [
  "What is Modern Portfolio Theory?",
  "What does the Sharpe ratio mean?",
  "Compare the three profiles",
  "Why is the high profile concentrated?",
  "Which stock had the highest historical volatility?",
  "Will NVDA go up next year?",
];

type Turn = { id: number; question: string | null; request: ExplainParams; result?: Schemas["ExplainResponse"]; error?: unknown };

function ModeBadge({ explanation }: { explanation: Schemas["Explanation"] }) {
  if (explanation.mode === "llm") return <Badge tone="primary"><Sparkles className="h-3 w-3" /> AI-selected facts</Badge>;
  if (explanation.mode === "fallback") {
    const reason = explanation.fallback_reason ?? "explanation_failed";
    return <Badge tone="warning" className="whitespace-normal">AI unavailable — {FALLBACK_TEXT[reason] ?? reason}</Badge>;
  }
  return <Badge>Deterministic (offline)</Badge>;
}

function AnswerCard({ turn }: { turn: Turn }) {
  const [sources, setSources] = useState(false);
  const explanation = turn.result?.explanation;
  return (
    <Card>
      <div className="flex items-start gap-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary-soft text-primary"><Bot className="h-4 w-4" /></span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-text">{turn.question ?? "Profile overview"}</p>
          {!turn.result && !turn.error && (
            <p className="mt-3 flex items-center gap-2 text-sm text-muted"><Loader2 className="h-4 w-4 animate-spin" /> Selecting grounded facts… this can take up to a minute.</p>
          )}
          {turn.error !== undefined && (
            <div className="mt-3">
              <ErrorCard error={turn.error instanceof ApiError && turn.error.code === "rate_limited"
                ? new Error("Too many AI requests — try again in a minute.")
                : turn.error instanceof DOMException ? new Error("Request cancelled.") : turn.error} />
            </div>
          )}
          {explanation && (
            <>
              <div className="mt-3 flex flex-wrap gap-2">
                <ModeBadge explanation={explanation} />
                {explanation.status === "unsupported" ? <Badge tone="warning">Outside the supported data</Badge>
                  : turn.question && explanation.question_answered ? <Badge tone="positive">Question answered</Badge>
                    : turn.question ? <Badge>General comparison (question not answered)</Badge> : null}
              </div>
              {explanation.answer ? (
                <>
                  <div className="mt-4 space-y-3 text-sm leading-relaxed text-slate-700">
                    {explanation.answer.split("\n\n").map((paragraph, index) => <p key={index}>{paragraph}</p>)}
                  </div>
                  <Footnote>
                    Written by the model from the facts below. Every figure in it appears verbatim in one of them;
                    an answer stating any other number is discarded.
                  </Footnote>
                </>
              ) : (
                <div className="mt-4 space-y-3 text-sm leading-relaxed text-slate-700">
                  {explanation.text.split("\n\n").map((paragraph, index) => <p key={index}>{paragraph}</p>)}
                </div>
              )}
              {explanation.answer_rejected && (
                <Note tone="warning" className="mt-3">
                  The model&rsquo;s wording was discarded because it stated a figure the verified facts do not
                  support. The facts themselves are shown unchanged.
                </Note>
              )}
              <button type="button" aria-expanded={sources} onClick={() => setSources(!sources)} className="mt-4 text-xs font-semibold text-primary hover:underline">
                {sources
                  ? "Hide grounding"
                  : `${explanation.answer ? "Show grounding" : "Sources"} (${explanation.fact_ids.length} facts${explanation.tool_calls.length ? `, ${explanation.tool_calls.length} lookups` : ""})`}
              </button>
              {sources && (
                <div className="mt-2 space-y-3 rounded-xl bg-surface-muted p-3 text-xs text-slate-700">
                  {explanation.answer && (
                    <div className="space-y-2 border-b border-border pb-3">
                      {explanation.text.split("\n\n").map((paragraph, index) => <p key={index}>{paragraph}</p>)}
                    </div>
                  )}
                  <p><span className="font-semibold">Facts:</span> {explanation.fact_ids.join(", ")}</p>
                  {explanation.tool_calls.length > 0 && (
                    <p><span className="font-semibold">Lookups:</span> {explanation.tool_calls.map((call) => `${call.name}(${JSON.stringify(call.arguments)})`).join(", ")}</p>
                  )}
                </div>
              )}
              <p className="mt-3 text-xs text-muted">
                {explanation.model ? `Model ${explanation.model} · ` : ""}{turn.result?.orchestration} orchestration · Every figure is rendered by Python from the frozen analysis.
              </p>
            </>
          )}
        </div>
      </div>
    </Card>
  );
}

export function ExplainerView() {
  const hydrated = useSettingsHydrated();
  if (!hydrated) return <PageSkeleton />;
  return <ExplainerBody />;
}

function ExplainerBody() {
  const settings = useSettings();
  const health = useHealth();
  const params = useSearchParams();
  const explain = useExplain();
  const llmAvailable = health.data?.llm_configured ?? false;
  const useLlm = settings.useLlm && llmAvailable;

  const prefillAmount = params.get("amount");
  const [profile, setProfile] = useState<ProfileName>(() => {
    const requested = params.get("profile");
    return PROFILE_NAMES.includes(requested as ProfileName) ? (requested as ProfileName) : settings.profile;
  });
  const [amount] = useState(() => (prefillAmount && isValidAmount(prefillAmount) ? prefillAmount : settings.amount));
  const [currency] = useState<Currency>(() => (params.get("currency") === "USD" ? "USD" : params.get("currency") === "INR" ? "INR" : settings.currency));
  const [includeAllocation, setIncludeAllocation] = useState(!!prefillAmount);
  const [emphasis, setEmphasis] = useState<Emphasis | "">("");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const nextId = useRef(1);
  const controller = useRef<AbortController | null>(null);
  const pending = turns.some((turn) => !turn.result && turn.error === undefined);

  const ask = (text: string | null) => {
    if (pending) return;
    const request: ExplainParams = {
      question: text, use_llm: useLlm, profile, currency, hide_dust: false,
      amount: includeAllocation && isValidAmount(amount) ? amount : null,
      preferences: emphasis ? { emphasis, max_volatility: null, min_expected_return: null } : null,
    };
    const id = nextId.current++;
    setTurns((current) => [{ id, question: text, request }, ...current]);
    controller.current = new AbortController();
    explain.mutate({ body: request, signal: controller.current.signal }, {
      onSuccess: (result) => setTurns((current) => current.map((t) => (t.id === id ? { ...t, result } : t))),
      onError: (error) => setTurns((current) => current.map((t) => (t.id === id ? { ...t, error } : t))),
    });
    if (text) setQuestion("");
  };

  const tooLong = question.length > 2000;
  return (
    <>
      <PageHeader title="AI Explainer" subtitle="Ask about the historical profiles. Answers are assembled from verified facts — never forecasts or advice." />
      <div className="grid gap-5 xl:grid-cols-[minmax(300px,1fr)_minmax(0,2.4fr)]">
        <div className="space-y-5">
          <Card>
            <CardHeader title="Context sent" subtitle="Only these inputs and verified analysis facts are used." />
            <div className="space-y-5">
              <div>
                <p className="mb-1.5 text-xs font-semibold text-slate-700">Profile</p>
                <Select label="Profile" className="w-full" value={profile} onChange={(e) => setProfile(e.target.value as ProfileName)}>
                  {PROFILE_NAMES.map((name) => <option key={name} value={name}>{PROFILE_LABELS[name]}</option>)}
                </Select>
              </div>
              <Switch label="Include my allocation" description={isValidAmount(amount) ? `${money(amount, currency)} in the ${PROFILE_LABELS[profile]} profile` : "Set a valid planned amount first"}
                checked={includeAllocation} disabled={!isValidAmount(amount)} onChange={setIncludeAllocation} />
              <div>
                <p className="mb-1.5 text-xs font-semibold text-slate-700">My emphasis (optional)</p>
                <Select label="Emphasis" className="w-full" value={emphasis} onChange={(e) => setEmphasis(e.target.value as Emphasis | "")}>
                  <option value="">Not specified</option>
                  {(Object.keys(EMPHASIS_LABELS) as Emphasis[]).map((value) => <option key={value} value={value}>{EMPHASIS_LABELS[value]}</option>)}
                </Select>
              </div>
            </div>
          </Card>
          <Card>
            <Switch label="AI mode" checked={useLlm} disabled={!llmAvailable} onChange={(useLlmValue) => settings.update({ useLlm: useLlmValue })}
              description={llmAvailable
                ? "Off: deterministic overview only. On: the model selects relevant facts for your question (sent to Groq)."
                : "Not configured on the server (needs GROQ_API_KEY and agent dependencies). Offline overviews still work."} />
          </Card>
        </div>

        <div className="space-y-5">
          <Card>
            <form onSubmit={(event) => { event.preventDefault(); if (question.trim() && !tooLong) ask(question); }}>
              <label htmlFor="question" className="text-sm font-semibold text-text">Your question</label>
              <textarea id="question" rows={3} value={question} disabled={!useLlm} onChange={(e) => setQuestion(e.target.value)}
                placeholder={useLlm ? "e.g. Why is the Growth profile more volatile?" : "Turn on AI mode to ask questions. You can still generate the overview."}
                className={clsx("mt-2 w-full resize-y rounded-xl border px-3 py-2.5 text-sm outline-none focus:border-primary disabled:bg-surface-muted",
                  tooLong ? "border-negative" : "border-border")} />
              <div className="mt-1 flex justify-between text-xs text-muted">
                <span>One question at a time; no chat memory.</span>
                <span className={tooLong ? "text-negative" : undefined}>{question.length}/2000</span>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button type="submit" variant="primary" disabled={!useLlm || !question.trim() || tooLong || pending}><Send className="h-4 w-4" /> Ask</Button>
                <Button type="button" disabled={pending} onClick={() => ask(null)}><Sparkles className="h-4 w-4" /> Generate overview</Button>
                {pending && <Button type="button" variant="ghost" onClick={() => controller.current?.abort()}><Square className="h-4 w-4" /> Cancel</Button>}
              </div>
            </form>
            {useLlm && (
              <div className="mt-4 flex flex-wrap gap-2">
                {SUGGESTIONS.map((suggestion) => (
                  <button key={suggestion} type="button" disabled={pending} onClick={() => ask(suggestion)}
                    className="rounded-full border border-border px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-surface-muted disabled:opacity-50">
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </Card>
          <Note>The model can only pick from facts Python computed from the frozen 2021–2025 data, plus fixed definitions of the method itself — so &ldquo;what does this mean&rdquo; questions are answerable. Questions about the future, live prices or personal circumstances are reported as unsupported.</Note>
          {turns.length === 0 ? (
            <EmptyState title="No answers yet">Generate the overview, or turn on AI mode and ask a question.</EmptyState>
          ) : turns.map((turn) => <AnswerCard key={turn.id} turn={turn} />)}
        </div>
      </div>
    </>
  );
}
