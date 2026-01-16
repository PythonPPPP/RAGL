import React from 'react'
import { useApp } from '../store/app'
import { api } from '../api/client'
import type { ChatAnswer, DatasetItem, PipelineItem } from '../api/types'
import { Badge, Button, Card, CardContent, CardHeader, Divider, Input, Textarea, Select, cn } from '../components/ui'
import { Bot, Brain, Loader2, Play, Sparkles, Timer, Coins, Hash } from 'lucide-react'

function SourceCard({
  rank,
  filename,
  score,
  text,
}: {
  rank: number
  filename: string
  score: number
  text: string
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold">
          [{rank}] <span className="text-white/70 font-medium">{filename}</span>
        </div>
        <Badge tone="neutral">score {score.toFixed(3)}</Badge>
      </div>
      <div className="text-xs text-white/60 mt-2 whitespace-pre-wrap leading-relaxed">
        {text.length > 600 ? text.slice(0, 600) + '…' : text}
      </div>
    </div>
  )
}

export function TheLab() {
  const { selectedProject } = useApp()
  const projectId = selectedProject?.id

  const [pipelines, setPipelines] = React.useState<PipelineItem[]>([])
  const [datasets, setDatasets] = React.useState<DatasetItem[]>([])
  const [pipelineId, setPipelineId] = React.useState<number | null>(null)
  const [datasetId, setDatasetId] = React.useState<number | null>(null)

  const [overrideModel, setOverrideModel] = React.useState('')
  const [question, setQuestion] = React.useState('')
  const [answer, setAnswer] = React.useState<ChatAnswer | null>(null)

  const [busy, setBusy] = React.useState(false)
  const [err, setErr] = React.useState<string | null>(null)
  const [runMsg, setRunMsg] = React.useState<string | null>(null)

  async function refresh() {
    if (!projectId) return
    const [ps, ds] = await Promise.all([api.listPipelines(projectId), api.listDatasets(projectId)])
    setPipelines(ps)
    setDatasets(ds)
    if (ps.length > 0 && (pipelineId === null || !ps.find((p) => p.id === pipelineId))) setPipelineId(ps[0].id)
    if (ds.length > 0 && (datasetId === null || !ds.find((d) => d.id === datasetId))) setDatasetId(ds[0].id)
  }

  React.useEffect(() => {
    refresh().catch(() => undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  if (!projectId) {
    return (
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">The Lab</div>
        </CardHeader>
        <CardContent>
          <Badge tone="warn">Select or create a project first.</Badge>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-1 space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold">Playground</div>
                <div className="text-sm text-white/60">Ask questions and inspect retrieval</div>
              </div>
              <Badge tone="neutral">Local engine</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {err ? (
              <Badge tone="bad">{err}</Badge>
            ) : null}

            <div>
              <div className="text-xs text-white/50 mb-1">Pipeline</div>
              <Select value={pipelineId ?? ''} onChange={(e) => setPipelineId(Number(e.target.value))}>
                {pipelines.length === 0 ? <option value="">—</option> : null}
                {pipelines.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </Select>
              <div className="text-xs text-white/40 mt-1">
                Build index in <span className="text-white/70">Architect</span> before asking.
              </div>
            </div>

            <div>
              <div className="text-xs text-white/50 mb-1">Override HF model (optional)</div>
              <Input
                value={overrideModel}
                onChange={(e) => setOverrideModel(e.target.value)}
                placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct"
              />
            </div>

            <div>
              <div className="text-xs text-white/50 mb-1">Question</div>
              <Textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask something about your documents…"
              />
            </div>

            <Button
              disabled={busy || !pipelineId}
              onClick={async () => {
                if (!pipelineId) return
                const q = question.trim()
                if (!q) {
                  setErr('Enter a question')
                  return
                }
                setBusy(true)
                setErr(null)
                setAnswer(null)
                try {
                  const out = await api.ask(projectId, pipelineId, q, overrideModel.trim() || undefined)
                  setAnswer(out)
                } catch (e: any) {
                  setErr(String(e?.message || e))
                } finally {
                  setBusy(false)
                }
              }}
            >
              {busy ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Play size={16} className="mr-2" />}
              Run
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold">Benchmark</div>
                <div className="text-sm text-white/60">Run evaluation on a dataset</div>
              </div>
              <Badge tone="neutral">Proxy-RAGAS</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <div className="text-xs text-white/50 mb-1">Dataset</div>
              <Select value={datasetId ?? ''} onChange={(e) => setDatasetId(Number(e.target.value))}>
                {datasets.length === 0 ? <option value="">—</option> : null}
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.count})
                  </option>
                ))}
              </Select>
            </div>
            <Button
              variant="secondary"
              disabled={busy || !pipelineId || !datasetId}
              onClick={async () => {
                if (!pipelineId || !datasetId) return
                setBusy(true)
                setRunMsg(null)
                setErr(null)
                try {
                  const r = await api.startEval(projectId, pipelineId, datasetId, 'Run from The Lab', true)
                  setRunMsg(`Started run #${String(r.run_id).padStart(3, '0')} (queued)`)
                } catch (e: any) {
                  setErr(String(e?.message || e))
                } finally {
                  setBusy(false)
                }
              }}
            >
              <Brain size={16} className="mr-2" />
              Run benchmark
            </Button>
            {runMsg ? <Badge tone="good">{runMsg}</Badge> : null}
            <div className="text-xs text-white/40">
              Open <span className="text-white/70">Insights</span> to see results and compare runs.
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="lg:col-span-2 space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Bot size={18} className="text-sky-200" />
                <div>
                  <div className="text-lg font-semibold">Answer</div>
                  <div className="text-sm text-white/60">Response with citations and sources</div>
                </div>
              </div>
              {answer?.timings?.total ? (
                <Badge tone="neutral">
                  <Timer size={14} className="mr-1" />
                  {answer.timings.total.toFixed(2)}s
                </Badge>
              ) : (
                <Badge tone="neutral">—</Badge>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {!answer ? (
              <div className="rounded-2xl border border-white/10 bg-white/4 p-6">
                <div className="text-lg font-semibold">Ready</div>
                <div className="text-sm text-white/60 mt-1">
                  Ask a question to see the retrieved context, latency breakdown, and the generated answer.
                </div>
              </div>
            ) : (
              <>
                <div className="rounded-2xl border border-white/10 bg-white/4 p-5">
                  <div className="text-xs text-white/50 mb-2">Generated</div>
                  <div className="whitespace-pre-wrap leading-relaxed text-white/90">{answer.answer}</div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                  <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
                    <div className="text-xs text-white/55">tokens</div>
                    <div className="mt-1 font-semibold text-white/85">
                      {answer.tokens ? `${answer.tokens.total ?? (answer.tokens.input + answer.tokens.output)}` : '—'}
                    </div>
                    <div className="text-xs text-white/40 mt-1">
                      in {answer.tokens?.input ?? 0} · out {answer.tokens?.output ?? 0}
                    </div>
                  </div>
                  {Object.entries(answer.timings)
                    .filter(([k]) => ['load_index', 'embed_query', 'retrieve', 'rerank', 'prompt', 'generate', 'total'].includes(k))
                    .map(([k, v]) => (
                      <div key={k} className="rounded-2xl border border-white/10 bg-white/4 p-4">
                        <div className="text-xs text-white/55">{k}</div>
                        <div className={cn('mt-1 font-semibold', k === 'total' ? 'text-white' : 'text-white/85')}>
                          {Number(v).toFixed(2)}s
                        </div>
                      </div>
                    ))}
                </div>

                <Divider />

                <div className="flex items-center justify-between">
                  <div className="text-sm font-semibold">Sources</div>
                  <Badge tone="neutral">
                    <Sparkles size={14} className="mr-1" />
                    {answer.sources.length} chunks
                  </Badge>
                </div>

                <div className="space-y-3">
                  {answer.sources.map((s) => (
                    <SourceCard key={`${s.doc_id}_${s.chunk_id}`} rank={s.rank} filename={s.filename} score={s.score} text={s.text} />
                  ))}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
