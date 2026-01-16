import React from 'react'
import { useApp } from '../store/app'
import { api } from '../api/client'
import type { PipelineItem, RunItem } from '../api/types'
import { Badge, Button, Card, CardContent, CardHeader, Divider, Select } from '../components/ui'
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, Tooltip, ResponsiveContainer } from 'recharts'
import { Activity, BarChart3, Sparkles } from 'lucide-react'

type MetricKey = 'faithfulness' | 'answer_relevancy' | 'context_precision' | 'context_recall' | 'latency_inv'

function toPct(delta: number) {
  const sign = delta >= 0 ? '+' : ''
  return `${sign}${Math.round(delta * 100)}%`
}

function safeRatio(cur: number, base: number): number {
  if (!isFinite(cur) || !isFinite(base) || base === 0) return 0
  return (cur - base) / base
}

function padRunId(id: number) {
  return `#${String(id).padStart(3, '0')}`
}

export function Insights() {
  const { selectedProject } = useApp()
  const projectId = selectedProject?.id

  const [runs, setRuns] = React.useState<RunItem[]>([])
  const [pipelines, setPipelines] = React.useState<PipelineItem[]>([])
  const [baselineId, setBaselineId] = React.useState<number | null>(null)
  const [currentId, setCurrentId] = React.useState<number | null>(null)
  const [msg, setMsg] = React.useState<string | null>(null)

  async function refresh() {
    if (!projectId) return
    const [rs, ps] = await Promise.all([api.listRuns(projectId), api.listPipelines(projectId)])
    setRuns(rs)
    setPipelines(ps)

    const done = rs.filter((r) => r.status === 'done' && typeof r.metrics?.ragas_score === 'number')
    const best = done.slice().sort((a, b) => (b.metrics.ragas_score ?? 0) - (a.metrics.ragas_score ?? 0))[0]
    const first = done.slice().sort((a, b) => a.id - b.id)[0]
    setCurrentId(best?.id ?? done[0]?.id ?? null)
    setBaselineId(first?.id ?? done[0]?.id ?? null)
  }

  React.useEffect(() => {
    refresh().catch((e) => setMsg(String(e?.message || e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  if (!projectId) {
    return (
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">Insights</div>
        </CardHeader>
        <CardContent>
          <Badge tone="warn">Select or create a project first.</Badge>
        </CardContent>
      </Card>
    )
  }

  const current = runs.find((r) => r.id === currentId) || null
  const baseline = runs.find((r) => r.id === baselineId) || null

  const getM = (r: RunItem | null, k: MetricKey) => {
    const v = r?.metrics?.[k]
    return typeof v === 'number' ? v : 0
  }

  const metricRows = [
    { key: 'faithfulness' as const, label: 'Faithfulness' },
    { key: 'answer_relevancy' as const, label: 'Answer Relevancy' },
    { key: 'context_precision' as const, label: 'Context Precision' },
    { key: 'context_recall' as const, label: 'Context Recall' },
    { key: 'latency_inv' as const, label: 'Latency (inv)' },
  ]

  const chartData = metricRows.map((m) => ({
    metric: m.label,
    Current: getM(current, m.key),
    Baseline: getM(baseline, m.key),
  }))

  const deltas = metricRows.map((m) => {
    const c = getM(current, m.key)
    const b = getM(baseline, m.key)
    return { key: m.key, label: m.label, cur: c, base: b, ratio: safeRatio(c, b) }
  })

  const bestMetric = deltas.slice().sort((a, b) => b.ratio - a.ratio)[0]
  const worstMetric = deltas.slice().sort((a, b) => a.ratio - b.ratio)[0]

  const ctxPrecDelta = safeRatio(getM(current, 'context_precision'), getM(baseline, 'context_precision'))
  const faithDelta = safeRatio(getM(current, 'faithfulness'), getM(baseline, 'faithfulness'))

  const currentScore = typeof current?.metrics?.ragas_score === 'number' ? current.metrics.ragas_score : 0
  const baseScore = typeof baseline?.metrics?.ragas_score === 'number' ? baseline.metrics.ragas_score : 0

  const bestRunId = runs
    .filter((r) => r.status === 'done' && typeof r.metrics?.ragas_score === 'number')
    .slice()
    .sort((a, b) => (b.metrics.ragas_score ?? 0) - (a.metrics.ragas_score ?? 0))[0]?.id

  const pipelineName = (pipelineId: number) => pipelines.find((p) => p.id === pipelineId)?.name || `Pipeline ${pipelineId}`

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-3xl font-bold tracking-tight">Benchmark Insights</div>
          <div className="text-white/60 mt-1">Compare runs, drill into metrics, and pick the best pipeline.</div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => refresh().catch(() => undefined)}>
            Refresh
          </Button>
          {msg ? <Badge tone="neutral">{msg}</Badge> : null}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold">RAGAS Metrics Comparison</div>
                <div className="text-sm text-white/60">Current vs baseline (proxy metrics)</div>
              </div>
              <div className="flex items-center gap-2">
                <div className="text-xs text-white/50">Current</div>
                <Select className="w-[220px]" value={currentId ?? ''} onChange={(e) => setCurrentId(Number(e.target.value))}>
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {padRunId(r.id)} · {pipelineName(r.pipeline_id)}
                    </option>
                  ))}
                </Select>
                <div className="text-xs text-white/50">Baseline</div>
                <Select className="w-[220px]" value={baselineId ?? ''} onChange={(e) => setBaselineId(Number(e.target.value))}>
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {padRunId(r.id)} · {pipelineName(r.pipeline_id)}
                    </option>
                  ))}
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {runs.length === 0 ? (
              <div className="rounded-2xl border border-white/10 bg-white/4 p-6">
                <div className="text-lg font-semibold">No runs yet</div>
                <div className="text-sm text-white/60 mt-1">
                  Go to <span className="text-white">The Lab</span> and run a benchmark.
                </div>
              </div>
            ) : (
              <div className="h-[360px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={chartData} outerRadius="75%">
                    <PolarGrid stroke="rgba(255,255,255,0.08)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255,255,255,0.65)', fontSize: 12 }} />
                    <PolarRadiusAxis domain={[0, 1]} tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: 'rgba(15,26,47,0.92)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12 }}
                      labelStyle={{ color: 'rgba(255,255,255,0.8)' }}
                    />
                    <Legend wrapperStyle={{ color: 'rgba(255,255,255,0.7)' }} />
                    <Radar name="Current Run" dataKey="Current" stroke="rgba(59,130,246,0.9)" fill="rgba(59,130,246,0.25)" />
                    <Radar name="Baseline" dataKey="Baseline" stroke="rgba(148,163,184,0.8)" fill="rgba(148,163,184,0.18)" />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold">Performance Analysis</div>
                <div className="text-sm text-white/60">What changed and why it matters</div>
              </div>
              <Badge tone="neutral">
                <BarChart3 size={14} className="mr-1" />
                Score
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-sm text-white/70 leading-relaxed">
              {current && baseline ? (
                <>
                  Current run <span className="text-white">{padRunId(current.id)}</span> shows a{' '}
                  <span className="text-white">{toPct(ctxPrecDelta)}</span> change in Context Precision compared to baseline.
                  Faithfulness changed by <span className="text-white">{toPct(faithDelta)}</span>. Use reranking to reduce hallucinations (but expect higher latency).
                </>
              ) : (
                <>Run a benchmark to see analysis.</>
              )}
            </div>

            <div className="grid grid-cols-1 gap-3">
              <div className="rounded-2xl border border-emerald-500/25 bg-emerald-500/10 p-4">
                <div className="text-xs text-emerald-200/80">BEST METRIC</div>
                <div className="mt-1 text-xl font-bold">{bestMetric?.label ?? '—'}</div>
                <div className="text-sm text-emerald-100/80 mt-1">
                  {bestMetric ? `${bestMetric.cur.toFixed(2)} (${toPct(bestMetric.ratio)})` : '—'}
                </div>
              </div>

              <div className="rounded-2xl border border-rose-500/25 bg-rose-500/10 p-4">
                <div className="text-xs text-rose-200/80">WORST METRIC</div>
                <div className="mt-1 text-xl font-bold">{worstMetric?.label ?? '—'}</div>
                <div className="text-sm text-rose-100/80 mt-1">
                  {worstMetric ? `${worstMetric.cur.toFixed(2)} (${toPct(worstMetric.ratio)})` : '—'}
                </div>
              </div>
            </div>

            <Divider />

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
                <div className="text-xs text-white/55">Current score</div>
                <div className="mt-1 text-2xl font-bold">{current ? currentScore.toFixed(2) : '—'}</div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
                <div className="text-xs text-white/55">Baseline score</div>
                <div className="mt-1 text-2xl font-bold">{baseline ? baseScore.toFixed(2) : '—'}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity size={18} className="text-white/70" />
              <div>
                <div className="text-lg font-semibold">Experiment History</div>
                <div className="text-sm text-white/60">All runs for this project</div>
              </div>
            </div>
            <Badge tone="neutral">{runs.length} runs</Badge>
          </div>
        </CardHeader>
        <CardContent>
          {runs.length === 0 ? (
            <div className="text-sm text-white/60">No runs yet.</div>
          ) : (
            <div className="overflow-hidden rounded-2xl border border-white/10">
              <table className="w-full text-sm">
                <thead className="bg-white/5 text-white/60">
                  <tr>
                    <th className="text-left px-4 py-3 font-medium">RUN ID</th>
                    <th className="text-left px-4 py-3 font-medium">CONFIG</th>
                    <th className="text-left px-4 py-3 font-medium">RAGAS SCORE</th>
                    <th className="text-left px-4 py-3 font-medium">STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {runs
                    .slice()
                    .sort((a, b) => b.id - a.id)
                    .map((r) => {
                      const score = typeof r.metrics?.ragas_score === 'number' ? r.metrics.ragas_score : null
                      const isBest = bestRunId === r.id
                      const tone = r.status === 'done' ? (isBest ? 'good' : 'neutral') : r.status === 'error' ? 'bad' : 'warn'
                      return (
                        <tr key={r.id} className="border-t border-white/10 hover:bg-white/3">
                          <td className="px-4 py-3 font-medium text-white/85">{padRunId(r.id)}</td>
                          <td className="px-4 py-3 text-white/70">{pipelineName(r.pipeline_id)}</td>
                          <td className="px-4 py-3">
                            {score === null ? (
                              <span className="text-white/40">—</span>
                            ) : (
                              <span className={isBest ? 'text-emerald-300 font-semibold' : 'text-white/80'}>{score.toFixed(2)}</span>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <Badge tone={tone as any}>{isBest ? 'Best' : r.status === 'done' ? 'Done' : r.status}</Badge>
                          </td>
                        </tr>
                      )
                    })}
                </tbody>
              </table>
            </div>
          )}

          <div className="mt-4 flex items-center gap-2 text-xs text-white/55">
            <Sparkles size={14} className="text-white/50" />
            <span>
              Tip: Run “Hybrid + Rerank” to improve faithfulness. If latency is too high, reduce top-k or rerank_top_n.
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
