import React from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type NodeProps,
} from 'reactflow'
import 'reactflow/dist/style.css'

import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type {
  BuildJobStatus,
  ChatAnswer,
  GenerateResult,
  PipelineConfig,
  PipelineDetail,
  PipelineItem,
  RetrieveDebug,
  SharePayload,
} from '../api/types'
import { useApp } from '../store/app'
import { Badge, Button, Card, CardContent, CardHeader, Divider, Input, Select, Textarea, cn } from '../components/ui'
import {
  Boxes,
  Braces,
  CheckCircle2,
  Copy,
  CopyPlus,
  Download,
  FlaskConical,
  LayoutGrid,
  Loader2,
  Network,
  Play,
  Plus,
  RefreshCcw,
  Save,
  Share2,
  Sparkles,
  Trash2,
  Wand2,
} from 'lucide-react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

type NodeKind =
  | 'reader'
  | 'chunker'
  | 'embed'
  | 'vectorstore'
  | 'query'
  | 'retriever'
  | 'mmr'
  | 'reranker'
  | 'prompt'
  | 'generator'
  | 'guardrails'

type MetricSample = {
  ts: number
  latency: number
  tokens_in: number
  tokens_out: number
}

const METRICS_MAX = 60

function encode64(obj: any): string {
  const json = JSON.stringify(obj)
  const utf8 = encodeURIComponent(json)
  // eslint-disable-next-line no-undef
  return btoa(unescape(utf8))
}
function decode64(s: string): any {
  // eslint-disable-next-line no-undef
  const json = decodeURIComponent(escape(atob(s)))
  return JSON.parse(json)
}

function defaultConfig(): PipelineConfig {
  return {
    name: 'New Pipeline',
    description: 'Draft pipeline',
    chunker: { chunk_size: 900, overlap: 120 },
    // Multilingual-friendly defaults (RU/EN)
    embedding_model: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    reranker_model: 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
    query: { mode: 'none', max_queries: 3 },
    context: { max_chars: 8000 },
    guardrails: { require_citations: true, min_citations: 1 },
    retriever: {
      mode: 'hybrid',
      top_k: 16,
      hybrid_alpha: 0.6,
      use_reranker: true,
      rerank_top_n: 6,
      min_score: 0.1,
      use_mmr: false,
      mmr_lambda: 0.65,
      mmr_k: 30,
    },
    generator: {
      // Small instruction model
      model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
      temperature: 0.1,
      max_new_tokens: 256,
      top_p: 0.9,
      system_prompt:
        'Ты — аккуратный ассистент для RAG. Отвечай только на основе контекста. Если данных нет — честно скажи, что не знаешь.',
    },
    ui: {
      nodes: {},
    },
  }
}

function defaultLayout(kind: NodeKind): { x: number; y: number } {
  // Three columns: Ingest, Retrieve, Generate
  const colX: Record<string, number> = { ingest: 120, retrieve: 520, generate: 920 }
  const rowY = {
    reader: 80,
    chunker: 200,
    embed: 320,
    vectorstore: 440,
    query: 120,
    retriever: 250,
    mmr: 360,
    reranker: 470,
    prompt: 140,
    generator: 300,
    guardrails: 460,
  } as Record<NodeKind, number>

  const col =
    kind === 'reader' || kind === 'chunker' || kind === 'embed' || kind === 'vectorstore'
      ? 'ingest'
      : kind === 'query' || kind === 'retriever' || kind === 'mmr' || kind === 'reranker'
        ? 'retrieve'
        : 'generate'

  return { x: colX[col], y: rowY[kind] ?? 80 }
}

function enabledNodeIds(cfg: PipelineConfig): NodeKind[] {
  const out: NodeKind[] = ['reader', 'chunker', 'embed', 'vectorstore', 'retriever', 'prompt', 'generator']
  // optional:
  if (cfg.query && cfg.query.mode !== 'none') out.splice(out.indexOf('retriever'), 0, 'query')
  if (cfg.retriever.use_mmr) out.splice(out.indexOf('retriever') + 1, 0, 'mmr')
  if (cfg.retriever.use_reranker) out.splice(out.indexOf('retriever') + 1 + (cfg.retriever.use_mmr ? 1 : 0), 0, 'reranker')
  if (cfg.guardrails?.require_citations) out.push('guardrails')
  return out
}

function graphFromConfig(cfg: PipelineConfig): { nodes: Node[]; edges: Edge[] } {
  const kinds = enabledNodeIds(cfg)
  const nodes: Node[] = kinds.map((k) => {
    const saved = cfg.ui?.nodes?.[k]
    const pos = saved && typeof saved.x === 'number' && typeof saved.y === 'number' ? { x: saved.x, y: saved.y } : defaultLayout(k)
    return {
      id: k,
      type: 'module',
      position: pos,
      data: { kind: k },
    }
  })

  const edges: Edge[] = []
  for (let i = 0; i < kinds.length - 1; i++) {
    const a = kinds[i]
    const b = kinds[i + 1]
    edges.push({ id: `${a}->${b}`, source: a, target: b, animated: true, style: { strokeWidth: 2 } })
  }
  return { nodes, edges }
}

function formatNodeTitle(k: NodeKind) {
  switch (k) {
    case 'reader':
      return { title: 'Reader', subtitle: 'Documents → text', icon: Boxes }
    case 'chunker':
      return { title: 'Chunker', subtitle: 'Split into chunks', icon: LayoutGrid }
    case 'embed':
      return { title: 'Embeddings', subtitle: 'Dense vectors', icon: Sparkles }
    case 'vectorstore':
      return { title: 'Index', subtitle: 'FAISS + BM25', icon: Network }
    case 'query':
      return { title: 'Query', subtitle: 'Rewrite / expand', icon: Wand2 }
    case 'retriever':
      return { title: 'Retriever', subtitle: 'Dense · BM25 · Hybrid', icon: Network }
    case 'mmr':
      return { title: 'MMR', subtitle: 'Diversity control', icon: Braces }
    case 'reranker':
      return { title: 'Re-ranker', subtitle: 'Cross-encoder', icon: Braces }
    case 'prompt':
      return { title: 'Prompt', subtitle: 'Context shaping', icon: Braces }
    case 'generator':
      return { title: 'LLM', subtitle: 'Generate answer', icon: FlaskConical }
    case 'guardrails':
      return { title: 'Guardrails', subtitle: 'Citations & rules', icon: CheckCircle2 }
  }
}

function ModuleNode({ id, data, selected }: NodeProps<any>) {
  const kind: NodeKind = data.kind
  const meta = formatNodeTitle(kind)
  const Icon = meta.icon

  const tone =
    kind === 'generator'
      ? 'border-fuchsia-400/30'
      : kind === 'retriever'
        ? 'border-sky-400/30'
        : kind === 'vectorstore'
          ? 'border-emerald-400/25'
          : 'border-white/15'

  return (
    <div
      className={cn(
        'w-[260px] rounded-3xl border bg-white/5 backdrop-blur px-4 py-3 shadow-[0_16px_50px_-25px_rgba(0,0,0,0.65)]',
        tone,
        selected ? 'ring-2 ring-white/15' : 'hover:bg-white/7 transition'
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className={cn('h-10 w-10 rounded-2xl grid place-items-center border', tone, 'bg-white/5')}>
            <Icon size={18} className="text-white/85" />
          </div>
          <div>
            <div className="text-sm font-semibold">{meta.title}</div>
            <div className="text-xs text-white/55 mt-0.5">{meta.subtitle}</div>
          </div>
        </div>
        <div className="text-xs text-white/35">{id}</div>
      </div>
      <div className="mt-3 flex items-center justify-between">
        <div className="text-xs text-white/45">Click to configure</div>
        <div className="h-2 w-2 rounded-full bg-white/20" />
      </div>
    </div>
  )
}

function useMetricsStorage(projectId: number | undefined, pipelineId: number | null) {
  const key = React.useMemo(() => {
    if (!projectId || !pipelineId) return null
    return `rag_arch_metrics:${projectId}:${pipelineId}`
  }, [projectId, pipelineId])

  const [samples, setSamples] = React.useState<MetricSample[]>([])

  React.useEffect(() => {
    if (!key) {
      setSamples([])
      return
    }
    try {
      const raw = localStorage.getItem(key)
      if (!raw) return
      const arr = JSON.parse(raw)
      if (Array.isArray(arr)) setSamples(arr)
    } catch {
      // ignore
    }
  }, [key])

  const add = React.useCallback(
    (s: MetricSample) => {
      if (!key) return
      setSamples((prev) => {
        const next = [...prev, s].slice(-METRICS_MAX)
        try {
          localStorage.setItem(key, JSON.stringify(next))
        } catch {
          // ignore
        }
        return next
      })
    },
    [key]
  )

  const clear = React.useCallback(() => {
    if (!key) return
    setSamples([])
    try {
      localStorage.removeItem(key)
    } catch {
      // ignore
    }
  }, [key])

  return { samples, add, clear }
}

function ArchitectInner() {
  const { selectedProject } = useApp()
  const projectId = selectedProject?.id

  const [searchParams, setSearchParams] = useSearchParams()

  const [pipelines, setPipelines] = React.useState<PipelineItem[]>([])
  const [templates, setTemplates] = React.useState<Record<string, any> | null>(null)

  const [pipelineId, setPipelineId] = React.useState<number | null>(null)
  const [pipelineDetail, setPipelineDetail] = React.useState<PipelineDetail | null>(null)

  const [cfg, setCfg] = React.useState<PipelineConfig>(defaultConfig())
  const [dirty, setDirty] = React.useState(false)
  const [selectedNode, setSelectedNode] = React.useState<NodeKind>('retriever')

  const [job, setJob] = React.useState<BuildJobStatus | null>(null)
  const [jobBusy, setJobBusy] = React.useState(false)

  const [busy, setBusy] = React.useState(false)
  const [err, setErr] = React.useState<string | null>(null)
  const [toast, setToast] = React.useState<string | null>(null)

  const [tab, setTab] = React.useState<'properties' | 'playground' | 'metrics' | 'share'>('properties')
  const [playMode, setPlayMode] = React.useState<'full' | 'retrieve' | 'generate' | 'compare'>('full')

  const [question, setQuestion] = React.useState('')
  const [retrieval, setRetrieval] = React.useState<RetrieveDebug | null>(null)
  const [answer, setAnswer] = React.useState<ChatAnswer | null>(null)
  const [genPrompt, setGenPrompt] = React.useState('')
  const [genResult, setGenResult] = React.useState<GenerateResult | null>(null)
  const [comparePipelineId, setComparePipelineId] = React.useState<number | null>(null)
  const [compareA, setCompareA] = React.useState<ChatAnswer | null>(null)
  const [compareB, setCompareB] = React.useState<ChatAnswer | null>(null)
  const [playBusy, setPlayBusy] = React.useState(false)

  const { samples, add: addSample, clear: clearSamples } = useMetricsStorage(projectId, pipelineId)

  const { nodes: initialNodes, edges: initialEdges } = React.useMemo(() => graphFromConfig(cfg), [cfg])
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  // keep reactflow graph synced with config
  React.useEffect(() => {
    const g = graphFromConfig(cfg)
    setNodes(g.nodes)
    setEdges(g.edges)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg])

  const nodeTypes = React.useMemo(() => ({ module: ModuleNode }), [])

  function updateCfg(next: PipelineConfig) {
    // persist node positions into cfg.ui.nodes
    const uiNodes: Record<string, any> = { ...(next.ui?.nodes || {}) }
    nodes.forEach((n) => {
      uiNodes[n.id] = { x: n.position.x, y: n.position.y }
    })
    next = { ...next, ui: { ...(next.ui || {}), nodes: uiNodes } }
    setCfg(next)
    setDirty(true)
  }

  async function refresh() {
    if (!projectId) return
    setErr(null)
    const [ps, ts] = await Promise.all([api.listPipelines(projectId), api.templates()])
    setPipelines(ps)
    setTemplates(ts)
    if (ps.length > 0 && (pipelineId === null || !ps.find((p) => p.id === pipelineId))) {
      setPipelineId(ps[0].id)
    }
  }

  // initial load per project
  React.useEffect(() => {
    refresh().catch((e) => setErr(String(e?.message || e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  // load pipeline detail when selected
  React.useEffect(() => {
    if (!projectId || !pipelineId) {
      setPipelineDetail(null)
      return
    }
    api
      .getPipeline(pipelineId)
      .then((d) => {
        setPipelineDetail(d)
        setCfg({ ...d.config, name: d.name, description: d.description })
        setDirty(false)
        setJob(null)
        setRetrieval(null)
        setAnswer(null)
        setGenResult(null)
        setCompareA(null)
        setCompareB(null)
        setPlayMode('full')
      })
      .catch((e) => setErr(String(e?.message || e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, pipelineId])

  // pick default compare pipeline
  React.useEffect(() => {
    if (pipelines.length === 0) {
      setComparePipelineId(null)
      return
    }
    // if current pipeline is not saved yet, compare against the first saved pipeline
    if (!pipelineId) {
      setComparePipelineId(pipelines[0]?.id ?? null)
      return
    }
    if (comparePipelineId && comparePipelineId !== pipelineId) return
    const alt = pipelines.find((p) => p.id !== pipelineId)?.id ?? null
    setComparePipelineId(alt)
  }, [pipelines, pipelineId, comparePipelineId])

  // import from URL params
  React.useEffect(() => {
    if (!projectId) return
    const shared = searchParams.get('share')
    const pip = searchParams.get('pipeline')
    if (!shared && !pip) return

    async function run() {
      try {
        let imported: PipelineConfig | null = null
        let meta: { name?: string; description?: string } = {}

        if (shared) {
          const p = (await api.getShare(shared)) as SharePayload
          imported = p.config
          meta = { name: p.name, description: p.description }
        } else if (pip) {
          const obj = decode64(pip)
          imported = obj?.config || obj
          meta = { name: obj?.name, description: obj?.description }
        }

        if (imported) {
          setPipelineId(null)
          setPipelineDetail(null)
          setCfg({ ...defaultConfig(), ...imported, name: meta.name || imported.name || 'Imported pipeline', description: meta.description || imported.description || '' })
          setDirty(true)
          setToast('Imported pipeline loaded (not saved yet).')
          setTimeout(() => setToast(null), 3500)
        }
      } catch (e: any) {
        setErr(String(e?.message || e))
      } finally {
        // clean url
        const next = new URLSearchParams(searchParams)
        next.delete('share')
        next.delete('pipeline')
        setSearchParams(next, { replace: true })
      }
    }

    run().catch(() => undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  // build job polling
  React.useEffect(() => {
    if (!pipelineId || !projectId || !job?.job_id) return
    if (job.status === 'done' || job.status === 'error') return

    const t = window.setInterval(async () => {
      try {
        const st = await api.buildIndexStatus(pipelineId)
        setJob(st)
        if (st.status === 'done') {
          setToast('Index built successfully.')
          setTimeout(() => setToast(null), 3500)
          const d = await api.getPipeline(pipelineId)
          setPipelineDetail(d)
        }
        if (st.status === 'error') {
          setErr(st.error || 'Index build failed')
        }
      } catch {
        // ignore
      }
    }, 1200)

    return () => window.clearInterval(t)
  }, [pipelineId, projectId, job?.job_id, job?.status])

  const unsaved = dirty || pipelineId === null

  async function savePipeline() {
    if (!projectId) return
    const name = (cfg.name || '').trim() || 'Pipeline'
    const description = cfg.description || ''

    setBusy(true)
    setErr(null)
    try {
      if (!pipelineId) {
        const created = await api.createPipeline(projectId, name, description, cfg)
        setPipelineId(created.id)
        await refresh()
        setToast('Pipeline created.')
      } else {
        await api.updatePipeline(pipelineId, projectId, name, description, cfg)
        setToast('Saved.')
      }
      setTimeout(() => setToast(null), 1800)
      setDirty(false)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setBusy(false)
    }
  }

  function newPipelineDraft() {
    setPipelineId(null)
    setPipelineDetail(null)
    setCfg(defaultConfig())
    setDirty(true)
    setSelectedNode('retriever')
    setTab('properties')
    setPlayMode('full')
    setJob(null)
    setRetrieval(null)
    setAnswer(null)
    setGenResult(null)
    setCompareA(null)
    setCompareB(null)
    setToast('New draft created (not saved).')
    setTimeout(() => setToast(null), 2000)
  }

  async function duplicatePipeline() {
    if (!projectId) return
    setBusy(true)
    setErr(null)
    try {
      // Prefer a server-side duplication when possible
      if (pipelineId) {
        const created = await api.duplicatePipeline(pipelineId)
        await refresh()
        setPipelineId(created.id)
        setToast('Pipeline duplicated.')
      } else {
        const name = ((cfg.name || 'Pipeline') + ' (copy)').trim()
        const created = await api.createPipeline(projectId, name, cfg.description || '', cfg)
        await refresh()
        setPipelineId(created.id)
        setToast('Pipeline duplicated.')
      }
      setTimeout(() => setToast(null), 1800)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setBusy(false)
    }
  }

  async function buildIndex() {
    if (!pipelineId) {
      setErr('Save the pipeline first')
      return
    }
    setJobBusy(true)
    setErr(null)
    try {
      const r = await api.buildIndexAsync(pipelineId)
      setJob({ job_id: r.job_id, status: r.status as any })
      setToast('Index build queued…')
      setTimeout(() => setToast(null), 2000)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setJobBusy(false)
    }
  }

  async function runRetrieve() {
    if (!projectId || !pipelineId) {
      setErr('Select a saved pipeline first')
      return
    }
    const q = question.trim()
    if (!q) {
      setErr('Enter a question')
      return
    }
    setPlayBusy(true)
    setPlayMode('retrieve')
    setErr(null)
    try {
      const r = await api.retrieveDebug(projectId, pipelineId, q)
      setRetrieval(r)
      setAnswer(null)
      setGenResult(null)
      setCompareA(null)
      setCompareB(null)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setPlayBusy(false)
    }
  }

  async function runPipeline() {
    if (!projectId || !pipelineId) {
      setErr('Select a saved pipeline first')
      return
    }
    const q = question.trim()
    if (!q) {
      setErr('Enter a question')
      return
    }
    setPlayBusy(true)
    setPlayMode('full')
    setErr(null)
    try {
      const a = await api.ask(projectId, pipelineId, q)
      setAnswer(a)
      setRetrieval(null)
      setGenResult(null)
      setCompareA(null)
      setCompareB(null)
      // store metrics sample
      const latency = Number(a.timings?.total || 0)
      const tokens_in = Number(a.tokens?.input || 0)
      const tokens_out = Number(a.tokens?.output || 0)
      addSample({ ts: Date.now(), latency, tokens_in, tokens_out })
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setPlayBusy(false)
    }
  }

  async function runGenerate() {
    const prompt = (genPrompt || '').trim()
    if (!prompt) {
      setErr('Enter a prompt')
      return
    }
    setPlayBusy(true)
    setPlayMode('generate')
    setErr(null)
    try {
      setAnswer(null)
      setRetrieval(null)
      setCompareA(null)
      setCompareB(null)
      const g = await api.generate(
        cfg.generator.model_id,
        prompt,
        cfg.generator.temperature,
        cfg.generator.max_new_tokens,
        cfg.generator.top_p
      )
      setGenResult(g)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setPlayBusy(false)
    }
  }

  async function runCompare() {
    if (!projectId || !pipelineId || !comparePipelineId) {
      setErr('Select two saved pipelines')
      return
    }
    const q = question.trim()
    if (!q) {
      setErr('Enter a question')
      return
    }
    setPlayBusy(true)
    setPlayMode('compare')
    setErr(null)
    try {
      setAnswer(null)
      setRetrieval(null)
      setGenResult(null)
      const [a, b] = await Promise.all([
        api.ask(projectId, pipelineId, q),
        api.ask(projectId, comparePipelineId, q),
      ])
      setCompareA(a)
      setCompareB(b)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setPlayBusy(false)
    }
  }

  function toggleModule(kind: NodeKind, enabled: boolean) {
    if (kind === 'query') {
      updateCfg({ ...cfg, query: { ...(cfg.query || { max_queries: 3 }), mode: enabled ? 'multi' : 'none' } })
      return
    }
    if (kind === 'mmr') {
      updateCfg({ ...cfg, retriever: { ...cfg.retriever, use_mmr: enabled } })
      return
    }
    if (kind === 'reranker') {
      updateCfg({ ...cfg, retriever: { ...cfg.retriever, use_reranker: enabled, rerank_top_n: enabled ? Math.max(1, cfg.retriever.rerank_top_n || 6) : 0 } })
      return
    }
    if (kind === 'guardrails') {
      updateCfg({ ...cfg, guardrails: { ...(cfg.guardrails || { min_citations: 1 }), require_citations: enabled, min_citations: cfg.guardrails?.min_citations ?? 1 } })
      return
    }
  }

  function applyTemplate(key: string) {
    if (!templates) return
    const t = templates[key]
    if (!t) return
    const config: PipelineConfig = {
      ...(defaultConfig() as any),
      ...(t as any),
      name: t.name,
      description: t.description,
    }
    updateCfg(config)
    setToast(`Template “${t.name}” loaded.`)
    setTimeout(() => setToast(null), 2500)
  }

  function autoLayout() {
    const uiNodes: Record<string, any> = { ...(cfg.ui?.nodes || {}) }
    enabledNodeIds(cfg).forEach((k) => {
      uiNodes[k] = defaultLayout(k)
    })
    updateCfg({ ...cfg, ui: { ...(cfg.ui || {}), nodes: uiNodes } })
    setToast('Layout reset.')
    setTimeout(() => setToast(null), 1500)
  }

  function exportJson() {
    const payload = { name: cfg.name, description: cfg.description, config: cfg }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${(cfg.name || 'pipeline').replace(/\s+/g, '_')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  async function copyShareLink(short = false) {
    try {
      let url = window.location.origin + window.location.pathname
      if (short) {
        const r = await api.createShare(cfg.name || 'Pipeline', cfg.description || '', cfg)
        url += `?share=${encodeURIComponent(r.code)}`
      } else {
        const packed = encode64({ name: cfg.name, description: cfg.description, config: cfg })
        url += `?pipeline=${encodeURIComponent(packed)}`
      }
      await navigator.clipboard.writeText(url)
      setToast('Link copied.')
      setTimeout(() => setToast(null), 1800)
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  const selectedMeta = formatNodeTitle(selectedNode)

  if (!projectId) {
    return (
      <Card>
        <CardHeader>
          <div className="text-lg font-semibold">Architect</div>
        </CardHeader>
        <CardContent>
          <Badge tone="warn">Select or create a project first.</Badge>
        </CardContent>
      </Card>
    )
  }

  const idxBadge = pipelineDetail?.index_ready ? (
    <Badge tone="good">Index ready</Badge>
  ) : (
    <Badge tone="neutral">Index not built</Badge>
  )

  return (
    <div className="space-y-5">
      {/* Top bar */}
      <div className="flex flex-col lg:flex-row lg:items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="h-11 w-11 rounded-2xl bg-white/6 border border-white/10 grid place-items-center">
            <Network size={18} className="text-white/85" />
          </div>
          <div>
            <div className="text-lg font-semibold flex items-center gap-2">
              Pipeline Studio
              {unsaved ? <Badge tone="warn">unsaved</Badge> : <Badge tone="good">saved</Badge>}
            </div>
            <div className="text-xs text-white/55">Drag modules, tweak parameters, build index, test instantly.</div>
          </div>
        </div>

        <div className="flex-1" />

        <div className="flex flex-wrap items-center gap-2">
          <Select
            value={pipelineId ?? ''}
            onChange={(e) => setPipelineId(e.target.value ? Number(e.target.value) : null)}
            className="w-[260px]"
          >
            {pipelines.length === 0 ? <option value="">— no pipelines —</option> : null}
            {pipelines.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </Select>

          <Button variant="secondary" disabled={busy} onClick={() => refresh().catch(() => undefined)}>
            <RefreshCcw size={16} className="mr-2" />
            Refresh
          </Button>

          <Button variant="secondary" disabled={busy} onClick={newPipelineDraft}>
            <Plus size={16} className="mr-2" />
            New
          </Button>

          <Button variant="secondary" disabled={busy || (!pipelineId && !cfg)} onClick={duplicatePipeline}>
            <CopyPlus size={16} className="mr-2" />
            Duplicate
          </Button>

          <Button disabled={busy} onClick={savePipeline}>
            {busy ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Save size={16} className="mr-2" />}
            Save
          </Button>

          <Button variant="secondary" disabled={jobBusy || !pipelineId} onClick={buildIndex}>
            {jobBusy ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Sparkles size={16} className="mr-2" />}
            Build index
          </Button>

          <Button variant="secondary" onClick={() => setTab('playground')}>
            <Play size={16} className="mr-2" />
            Test
          </Button>

          <Button variant="ghost" onClick={() => copyShareLink(false)}>
            <Share2 size={16} className="mr-2" />
            Share
          </Button>
        </div>
      </div>

      {err ? (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-100">
          {err}
        </div>
      ) : null}

      {toast ? (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/80">{toast}</div>
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Library */}
        <div className="lg:col-span-3 space-y-5">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-lg font-semibold">Library</div>
                  <div className="text-sm text-white/60">Optional modules and templates</div>
                </div>
                {idxBadge}
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-xs text-white/45">Templates</div>
              <div className="grid grid-cols-1 gap-2">
                {templates ? (
                  <>
                    <Button variant="secondary" onClick={() => applyTemplate('naive')}>
                      <Sparkles size={16} className="mr-2" />
                      Naive (Fast)
                    </Button>
                    <Button variant="secondary" onClick={() => applyTemplate('advanced')}>
                      <Sparkles size={16} className="mr-2" />
                      Hybrid + Rerank
                    </Button>
                    <Button variant="secondary" onClick={() => applyTemplate('long_context')}>
                      <Sparkles size={16} className="mr-2" />
                      Long-context
                    </Button>
                    <Button variant="secondary" onClick={() => applyTemplate('diverse')}>
                      <Sparkles size={16} className="mr-2" />
                      Diverse (MMR)
                    </Button>
                  </>
                ) : (
                  <Badge tone="neutral">Loading…</Badge>
                )}
              </div>

              <Divider />

              <div className="text-xs text-white/45">Optional modules</div>
              <div className="space-y-2">
                <ToggleRow
                  title="Query expansion"
                  desc="Generate extra query variants"
                  checked={!!cfg.query && cfg.query.mode !== 'none'}
                  onChange={(v) => toggleModule('query', v)}
                />
                <ToggleRow
                  title="MMR"
                  desc="Increase diversity in retrieval"
                  checked={!!cfg.retriever.use_mmr}
                  onChange={(v) => toggleModule('mmr', v)}
                />
                <ToggleRow
                  title="Re-ranker"
                  desc="Cross-encoder reranking"
                  checked={!!cfg.retriever.use_reranker}
                  onChange={(v) => toggleModule('reranker', v)}
                />
                <ToggleRow
                  title="Guardrails"
                  desc="Enforce citations"
                  checked={!!cfg.guardrails?.require_citations}
                  onChange={(v) => toggleModule('guardrails', v)}
                />
              </div>

              <Divider />

              <div className="flex items-center gap-2">
                <Button variant="secondary" onClick={autoLayout}>
                  <LayoutGrid size={16} className="mr-2" />
                  Auto layout
                </Button>
                <Button variant="ghost" onClick={exportJson}>
                  <Download size={16} className="mr-2" />
                  Export
                </Button>
              </div>

              {pipelineId ? (
                <div className="pt-2">
                  <Button
                    variant="ghost"
                    disabled={busy}
                    onClick={async () => {
                      if (!pipelineId) return
                      if (!window.confirm('Delete this pipeline? This cannot be undone.')) return
                      setBusy(true)
                      setErr(null)
                      try {
                        await api.deletePipeline(pipelineId, false)
                        setPipelineId(null)
                        setPipelineDetail(null)
                        setCfg(defaultConfig())
                        await refresh()
                        setToast('Pipeline deleted.')
                        setTimeout(() => setToast(null), 2000)
                      } catch (e: any) {
                        setErr(String(e?.message || e))
                      } finally {
                        setBusy(false)
                      }
                    }}
                  >
                    <Trash2 size={16} className="mr-2" />
                    Delete pipeline
                  </Button>
                </div>
              ) : null}
            </CardContent>
          </Card>

          {job ? (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-lg font-semibold">Index build</div>
                    <div className="text-sm text-white/60">Background job</div>
                  </div>
                  <Badge tone={job.status === 'done' ? 'good' : job.status === 'error' ? 'bad' : 'neutral'}>
                    {job.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="text-xs text-white/60">job_id</div>
                <div className="text-xs text-white/75 break-all">{job.job_id}</div>
                {job.error ? <Badge tone="bad">{job.error}</Badge> : null}
                {job.result?.meta ? (
                  <div className="text-xs text-white/60">
                    chunks: <span className="text-white/80">{job.result.meta?.chunks ?? '—'}</span> · docs:{' '}
                    <span className="text-white/80">{job.result.meta?.docs ?? '—'}</span>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ) : null}
        </div>

        {/* Center: Canvas */}
        <div className="lg:col-span-6">
          <Card className="h-[740px]">
            <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold">Canvas</div>
                <div className="text-xs text-white/55">Drag nodes. Select a module to edit settings.</div>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="secondary" onClick={() => setTab('properties')}>
                  <Braces size={16} className="mr-2" />
                  Inspector
                </Button>
                <Button variant="secondary" onClick={() => setTab('playground')}>
                  <FlaskConical size={16} className="mr-2" />
                  Playground
                </Button>
              </div>
            </div>

            <div className="h-[calc(740px-64px)]">
              <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeClick={(_, n) => {
                  setSelectedNode(n.id as NodeKind)
                  setTab('properties')
                }}
                onNodeDragStop={() => {
                  // persist positions
                  const uiNodes: Record<string, any> = { ...(cfg.ui?.nodes || {}) }
                  nodes.forEach((n) => {
                    uiNodes[n.id] = { x: n.position.x, y: n.position.y }
                  })
                  setCfg((prev) => ({ ...prev, ui: { ...(prev.ui || {}), nodes: uiNodes } }))
                  setDirty(true)
                }}
                fitView
              >
                <Background gap={24} size={1} />
                <Controls />
                <MiniMap pannable zoomable />
              </ReactFlow>
            </div>
          </Card>
        </div>

        {/* Right: Inspector */}
        <div className="lg:col-span-3 space-y-5">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-lg font-semibold">Inspector</div>
                  <div className="text-sm text-white/60">
                    <span className="text-white/85">{selectedMeta.title}</span> · {selectedMeta.subtitle}
                  </div>
                </div>
                <div className="inline-flex rounded-2xl bg-white/5 border border-white/10 p-1">
                  {[
                    { k: 'properties', label: 'Props' },
                    { k: 'playground', label: 'Test' },
                    { k: 'metrics', label: 'Metrics' },
                    { k: 'share', label: 'Share' },
                  ].map((t) => (
                    <button
                      key={t.k}
                      className={cn(
                        'px-3 h-9 rounded-xl text-xs transition',
                        tab === (t.k as any) ? 'bg-white/10 text-white' : 'text-white/60 hover:bg-white/5 hover:text-white'
                      )}
                      onClick={() => setTab(t.k as any)}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              {tab === 'properties' ? (
                <PropertiesPanel cfg={cfg} selected={selectedNode} onCfg={(c) => updateCfg(c)} />
              ) : null}

              {tab === 'playground' ? (
                <div className="space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold">Playground</div>
                      <div className="text-xs text-white/55">Test retrieval, generator, full RAG run — or compare pipelines.</div>
                    </div>
                    <div className="inline-flex rounded-2xl bg-white/5 border border-white/10 p-1">
                      {[
                        { k: 'full' as const, label: 'Full' },
                        { k: 'retrieve' as const, label: 'Retrieve' },
                        { k: 'generate' as const, label: 'Generate' },
                        { k: 'compare' as const, label: 'Compare' },
                      ].map((m) => (
                        <button
                          key={m.k}
                          className={cn(
                            'px-3 h-9 rounded-xl text-xs transition',
                            playMode === m.k ? 'bg-white/10 text-white' : 'text-white/60 hover:bg-white/5 hover:text-white'
                          )}
                          onClick={() => setPlayMode(m.k)}
                        >
                          {m.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {playMode !== 'generate' ? (
                    <Textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask something about your documents…" />
                  ) : (
                    <Textarea value={genPrompt} onChange={(e) => setGenPrompt(e.target.value)} placeholder="Paste a raw prompt to test the LLM generator…" />
                  )}

                  {playMode === 'compare' ? (
                    <div className="space-y-2">
                      <div className="text-xs text-white/55">Compare with</div>
                      <Select
                        value={comparePipelineId ?? ''}
                        onChange={(e) => setComparePipelineId(e.target.value ? Number(e.target.value) : null)}
                      >
                        {comparePipelineId === null ? <option value="">— select —</option> : null}
                        {pipelines
                          .filter((p) => p.id !== pipelineId)
                          .map((p) => (
                            <option key={p.id} value={p.id}>
                              {p.name}
                            </option>
                          ))}
                      </Select>
                      {pipelines.filter((p) => p.id !== pipelineId).length === 0 ? (
                        <div className="text-xs text-white/50">Create a second pipeline to enable comparison.</div>
                      ) : null}
                    </div>
                  ) : null}

                  {playMode === 'generate' ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-1 gap-2">
                        <Field label="HF model">
                          <Input
                            value={cfg.generator.model_id}
                            onChange={(e) => updateCfg({ ...cfg, generator: { ...cfg.generator, model_id: e.target.value } })}
                            placeholder="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                          />
                        </Field>
                        <div className="grid grid-cols-3 gap-2">
                          <Field label="Temp">
                            <Input
                              type="number"
                              value={cfg.generator.temperature}
                              onChange={(e) => updateCfg({ ...cfg, generator: { ...cfg.generator, temperature: Number(e.target.value || 0.1) } })}
                            />
                          </Field>
                          <Field label="Max new">
                            <Input
                              type="number"
                              value={cfg.generator.max_new_tokens}
                              onChange={(e) => updateCfg({ ...cfg, generator: { ...cfg.generator, max_new_tokens: Math.max(1, Number(e.target.value || 256)) } })}
                            />
                          </Field>
                          <Field label="Top-p">
                            <Input
                              type="number"
                              value={cfg.generator.top_p}
                              onChange={(e) => updateCfg({ ...cfg, generator: { ...cfg.generator, top_p: Math.max(0.1, Math.min(1, Number(e.target.value || 0.9))) } })}
                            />
                          </Field>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="text-xs text-white/55">Tip: use this to sanity-check a new LLM before saving.</div>
                        <Button
                          variant="ghost"
                          onClick={() => setGenPrompt('You are a helpful assistant. Reply in 5 bullet points.\n\nTopic: RAG evaluation metrics')}
                        >
                          <Sparkles size={16} className="mr-2" />
                          Insert example
                        </Button>
                      </div>

                      <Button disabled={playBusy} onClick={runGenerate}>
                        {playBusy ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Wand2 size={16} className="mr-2" />}
                        Generate
                      </Button>

                      {genResult ? (
                        <div className="space-y-3">
                          <Divider />
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-semibold">Output</div>
                            <Badge tone="neutral">tokens {genResult.tokens.input + genResult.tokens.output}</Badge>
                          </div>
                          <div className="rounded-2xl border border-white/10 bg-white/4 p-3 text-sm whitespace-pre-wrap leading-relaxed">
                            {genResult.text}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  {playMode !== 'generate' ? (
                    <div className="flex items-center gap-2">
                      {playMode !== 'compare' ? (
                        <Button variant="secondary" disabled={playBusy} onClick={runRetrieve}>
                          {playBusy ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Wand2 size={16} className="mr-2" />}
                          Retrieve
                        </Button>
                      ) : null}
                      {playMode === 'compare' ? (
                        <Button disabled={playBusy} onClick={runCompare}>
                          {playBusy ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Play size={16} className="mr-2" />}
                          Compare
                        </Button>
                      ) : (
                        <Button disabled={playBusy} onClick={runPipeline}>
                          {playBusy ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Play size={16} className="mr-2" />}
                          Run
                        </Button>
                      )}
                    </div>
                  ) : null}

                  {(retrieval || answer || compareA || compareB) ? <Divider /> : null}

                  {playMode === 'retrieve' && retrieval ? (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold">Retrieval</div>
                        <Badge tone="neutral">{retrieval.sources.length} chunks</Badge>
                      </div>
                      <div className="text-xs text-white/55">Queries</div>
                      <div className="flex flex-wrap gap-2">
                        {retrieval.queries.map((q, i) => (
                          <Badge key={i} tone="neutral">
                            {q}
                          </Badge>
                        ))}
                      </div>
                      <details className="rounded-2xl border border-white/10 bg-white/4 p-3">
                        <summary className="cursor-pointer text-xs text-white/70">Context preview</summary>
                        <div className="mt-2 text-xs text-white/70 whitespace-pre-wrap max-h-[240px] overflow-auto">{retrieval.context_preview}</div>
                      </details>
                    </div>
                  ) : null}

                  {playMode === 'full' && answer ? (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold">Answer</div>
                        <Badge tone="neutral">{answer.timings?.total?.toFixed(2)}s</Badge>
                      </div>
                      <div className="rounded-2xl border border-white/10 bg-white/4 p-3 text-sm whitespace-pre-wrap leading-relaxed">{answer.answer}</div>
                      <div className="grid grid-cols-3 gap-2">
                        <MiniStat label="tokens" value={answer.tokens ? String(answer.tokens.total ?? answer.tokens.input + answer.tokens.output) : '—'} />
                        <MiniStat label="ctx chars" value={answer.context_chars ? String(answer.context_chars) : '—'} />
                        <MiniStat label="cost" value={answer.cost_usd != null ? `$${answer.cost_usd.toFixed(4)}` : '—'} />
                      </div>
                      <details className="rounded-2xl border border-white/10 bg-white/4 p-3">
                        <summary className="cursor-pointer text-xs text-white/70">Sources ({answer.sources?.length ?? 0})</summary>
                        <div className="mt-2 space-y-2">
                          {(answer.sources || []).slice(0, 6).map((s) => (
                            <div key={s.rank} className="rounded-xl border border-white/10 bg-white/3 p-2">
                              <div className="text-xs text-white/80">[{s.rank}] {s.filename} · score {Number(s.score).toFixed(3)}</div>
                              <div className="text-xs text-white/60 mt-1 whitespace-pre-wrap">{(s.text || '').slice(0, 240)}{(s.text || '').length > 240 ? '…' : ''}</div>
                            </div>
                          ))}
                        </div>
                      </details>
                      {answer.prompt_preview ? (
                        <details className="rounded-2xl border border-white/10 bg-white/4 p-3">
                          <summary className="cursor-pointer text-xs text-white/70">Prompt preview</summary>
                          <div className="mt-2 text-xs text-white/70 whitespace-pre-wrap max-h-[240px] overflow-auto">{answer.prompt_preview}</div>
                        </details>
                      ) : null}
                    </div>
                  ) : null}

                  {playMode === 'compare' && compareA && compareB ? (
                    <div className="grid grid-cols-1 gap-3">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-semibold">A · {pipelineDetail?.name || 'Current'}</div>
                            <Badge tone="neutral">{compareA.timings?.total?.toFixed(2)}s</Badge>
                          </div>
                          <div className="mt-3 text-sm whitespace-pre-wrap leading-relaxed">{compareA.answer}</div>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-semibold">B · {pipelines.find((p) => p.id === comparePipelineId)?.name || 'Compare'}</div>
                            <Badge tone="neutral">{compareB.timings?.total?.toFixed(2)}s</Badge>
                          </div>
                          <div className="mt-3 text-sm whitespace-pre-wrap leading-relaxed">{compareB.answer}</div>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <MiniStat label="A tokens" value={compareA.tokens ? String(compareA.tokens.total ?? compareA.tokens.input + compareA.tokens.output) : '—'} />
                        <MiniStat label="B tokens" value={compareB.tokens ? String(compareB.tokens.total ?? compareB.tokens.input + compareB.tokens.output) : '—'} />
                        <MiniStat
                          label="Δ latency"
                          value={`${(Number(compareB.timings?.total || 0) - Number(compareA.timings?.total || 0)).toFixed(2)}s`}
                        />
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}

              {tab === 'metrics' ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-semibold">Local metrics</div>
                      <div className="text-xs text-white/55">Stored in your browser for this pipeline</div>
                    </div>
                    <Button variant="ghost" onClick={clearSamples}>
                      Clear
                    </Button>
                  </div>

                  {samples.length === 0 ? (
                    <div className="rounded-2xl border border-white/10 bg-white/4 p-4 text-sm text-white/70">
                      Run a few queries in the Playground to populate latency & token charts.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="h-[140px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={samples.map((s) => ({
                              ...s,
                              t: new Date(s.ts).toLocaleTimeString(),
                              tokens: (s.tokens_in || 0) + (s.tokens_out || 0),
                            }))}
                          >
                            <XAxis dataKey="t" hide />
                            <YAxis hide />
                            <Tooltip contentStyle={{ background: 'rgba(12,18,32,0.9)', border: '1px solid rgba(255,255,255,0.1)' }} />
                            <Line type="monotone" dataKey="latency" dot={false} strokeWidth={2} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="h-[140px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={samples.map((s) => ({
                              ...s,
                              t: new Date(s.ts).toLocaleTimeString(),
                              tokens: (s.tokens_in || 0) + (s.tokens_out || 0),
                            }))}
                          >
                            <XAxis dataKey="t" hide />
                            <YAxis hide />
                            <Tooltip contentStyle={{ background: 'rgba(12,18,32,0.9)', border: '1px solid rgba(255,255,255,0.1)' }} />
                            <Line type="monotone" dataKey="tokens" dot={false} strokeWidth={2} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="grid grid-cols-3 gap-2">
                        <MiniStat label="samples" value={String(samples.length)} />
                        <MiniStat label="avg latency" value={`${(samples.reduce((a, b) => a + b.latency, 0) / samples.length).toFixed(2)}s`} />
                        <MiniStat
                          label="avg tokens"
                          value={String(
                            Math.round(
                              samples.reduce((a, b) => a + b.tokens_in + b.tokens_out, 0) / samples.length
                            )
                          )}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ) : null}

              {tab === 'share' ? (
                <div className="space-y-3">
                  <div className="text-sm font-semibold">Share & export</div>
                  <div className="grid grid-cols-1 gap-2">
                    <Button variant="secondary" onClick={() => copyShareLink(false)}>
                      <Copy size={16} className="mr-2" />
                      Copy JSON link
                    </Button>
                    <Button variant="secondary" onClick={() => copyShareLink(true)}>
                      <Share2 size={16} className="mr-2" />
                      Copy short link
                    </Button>
                    <Button variant="ghost" onClick={exportJson}>
                      <Download size={16} className="mr-2" />
                      Export JSON file
                    </Button>
                  </div>

                  <Divider />

                  <div className="text-xs text-white/55">
                    Tip: a short link stores the config in local SQLite (backend). The JSON link encodes the config directly
                    into the URL.
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function ToggleRow({
  title,
  desc,
  checked,
  onChange,
}: {
  title: string
  desc: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/4 p-3 flex items-center justify-between">
      <div>
        <div className="text-sm font-semibold">{title}</div>
        <div className="text-xs text-white/55 mt-0.5">{desc}</div>
      </div>
      <button
        className={cn(
          'h-8 w-14 rounded-full border transition relative',
          checked ? 'bg-emerald-500/20 border-emerald-500/30' : 'bg-white/5 border-white/10'
        )}
        onClick={() => onChange(!checked)}
        aria-label={title}
      >
        <span
          className={cn(
            'absolute top-1 h-6 w-6 rounded-full bg-white/75 transition',
            checked ? 'left-7' : 'left-1'
          )}
        />
      </button>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/4 p-3">
      <div className="text-xs text-white/55">{label}</div>
      <div className="mt-1 font-semibold text-white/90">{value}</div>
    </div>
  )
}

function PropertiesPanel({
  cfg,
  selected,
  onCfg,
}: {
  cfg: PipelineConfig
  selected: NodeKind
  onCfg: (cfg: PipelineConfig) => void
}) {
  const name = (cfg.name || '').trim()

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="text-xs text-white/55">Pipeline name</div>
        <Input value={name} onChange={(e) => onCfg({ ...cfg, name: e.target.value })} placeholder="Pipeline name" />
        <div className="text-xs text-white/55">Description</div>
        <Input value={cfg.description || ''} onChange={(e) => onCfg({ ...cfg, description: e.target.value })} placeholder="Optional" />
      </div>

      <Divider />

      {/* Node-specific props */}
      {selected === 'chunker' ? (
        <div className="space-y-3">
          <Field label="Chunk size">
            <Input
              type="number"
              value={cfg.chunker.chunk_size}
              onChange={(e) => onCfg({ ...cfg, chunker: { ...cfg.chunker, chunk_size: Math.max(200, Number(e.target.value || 900)) } })}
            />
          </Field>
          <Field label="Overlap">
            <Input
              type="number"
              value={cfg.chunker.overlap}
              onChange={(e) => onCfg({ ...cfg, chunker: { ...cfg.chunker, overlap: Math.max(0, Number(e.target.value || 120)) } })}
            />
          </Field>
        </div>
      ) : null}

      {selected === 'embed' ? (
        <div className="space-y-3">
          <Field label="Embedding model">
            <Input value={cfg.embedding_model} onChange={(e) => onCfg({ ...cfg, embedding_model: e.target.value })} />
          </Field>
          <div className="text-xs text-white/45">SentenceTransformers model (HF id)</div>
        </div>
      ) : null}

      {selected === 'query' ? (
        <div className="space-y-3">
          <Field label="Mode">
            <Select
              value={cfg.query?.mode || 'none'}
              onChange={(e) => onCfg({ ...cfg, query: { ...(cfg.query || { max_queries: 3 }), mode: e.target.value as any } })}
            >
              <option value="none">none</option>
              <option value="keywords">keywords</option>
              <option value="multi">multi</option>
            </Select>
          </Field>
          <Field label="Max queries">
            <Input
              type="number"
              value={cfg.query?.max_queries ?? 3}
              onChange={(e) => onCfg({ ...cfg, query: { ...(cfg.query || { mode: 'multi' }), max_queries: Math.max(1, Number(e.target.value || 3)) } })}
            />
          </Field>
        </div>
      ) : null}

      {selected === 'retriever' || selected === 'mmr' || selected === 'reranker' ? (
        <div className="space-y-3">
          <Field label="Retriever mode">
            <Select value={cfg.retriever.mode} onChange={(e) => onCfg({ ...cfg, retriever: { ...cfg.retriever, mode: e.target.value as any } })}>
              <option value="dense">dense</option>
              <option value="bm25">bm25</option>
              <option value="hybrid">hybrid</option>
            </Select>
          </Field>
          <Field label="Top K">
            <Input
              type="number"
              value={cfg.retriever.top_k}
              onChange={(e) => onCfg({ ...cfg, retriever: { ...cfg.retriever, top_k: Math.max(1, Number(e.target.value || 16)) } })}
            />
          </Field>
          {cfg.retriever.mode === 'hybrid' ? (
            <Field label="Hybrid α (dense weight)">
              <Input
                type="number"
                value={cfg.retriever.hybrid_alpha ?? 0.6}
                onChange={(e) => {
                  const v = Math.max(0, Math.min(1, Number(e.target.value || 0.6)))
                  onCfg({ ...cfg, retriever: { ...cfg.retriever, hybrid_alpha: v } })
                }}
              />
            </Field>
          ) : null}
          <Field label="Min score (optional)">
            <Input
              type="number"
              value={cfg.retriever.min_score ?? ''}
              onChange={(e) => {
                const raw = e.target.value
                onCfg({ ...cfg, retriever: { ...cfg.retriever, min_score: raw === '' ? null : Number(raw) } })
              }}
              placeholder="0.1"
            />
          </Field>

          <Divider />

          <div className="rounded-2xl border border-white/10 bg-white/4 p-3 space-y-3">
            <div className="text-sm font-semibold">MMR</div>
            <Field label="Enabled">
              <Select
                value={cfg.retriever.use_mmr ? 'yes' : 'no'}
                onChange={(e) => onCfg({ ...cfg, retriever: { ...cfg.retriever, use_mmr: e.target.value === 'yes' } })}
              >
                <option value="yes">yes</option>
                <option value="no">no</option>
              </Select>
            </Field>
            {cfg.retriever.use_mmr ? (
              <>
                <Field label="λ (relevance vs diversity)">
                  <Input
                    type="number"
                    value={cfg.retriever.mmr_lambda ?? 0.65}
                    onChange={(e) => {
                      const v = Math.max(0, Math.min(1, Number(e.target.value || 0.65)))
                      onCfg({ ...cfg, retriever: { ...cfg.retriever, mmr_lambda: v } })
                    }}
                  />
                </Field>
                <Field label="Candidate pool">
                  <Input
                    type="number"
                    value={cfg.retriever.mmr_k ?? 30}
                    onChange={(e) => onCfg({ ...cfg, retriever: { ...cfg.retriever, mmr_k: Math.max(10, Number(e.target.value || 30)) } })}
                  />
                </Field>
              </>
            ) : null}
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/4 p-3 space-y-3">
            <div className="text-sm font-semibold">Re-ranker</div>
            <Field label="Enabled">
              <Select
                value={cfg.retriever.use_reranker ? 'yes' : 'no'}
                onChange={(e) => {
                  const yes = e.target.value === 'yes'
                  onCfg({ ...cfg, retriever: { ...cfg.retriever, use_reranker: yes, rerank_top_n: yes ? Math.max(1, cfg.retriever.rerank_top_n || 6) : 0 } })
                }}
              >
                <option value="yes">yes</option>
                <option value="no">no</option>
              </Select>
            </Field>
            {cfg.retriever.use_reranker ? (
              <>
                <Field label="Reranker model">
                  <Input value={cfg.reranker_model} onChange={(e) => onCfg({ ...cfg, reranker_model: e.target.value })} />
                </Field>
                <Field label="rerank_top_n">
                  <Input
                    type="number"
                    value={cfg.retriever.rerank_top_n}
                    onChange={(e) => onCfg({ ...cfg, retriever: { ...cfg.retriever, rerank_top_n: Math.max(1, Number(e.target.value || 6)) } })}
                  />
                </Field>
              </>
            ) : null}
          </div>
        </div>
      ) : null}

      {selected === 'prompt' ? (
        <div className="space-y-3">
          <Field label="Context max chars">
            <Input
              type="number"
              value={cfg.context?.max_chars ?? 8000}
              onChange={(e) => onCfg({ ...cfg, context: { ...(cfg.context || {}), max_chars: Math.max(1000, Number(e.target.value || 8000)) } })}
            />
          </Field>
          <Field label="System prompt">
            <Textarea
              value={cfg.generator.system_prompt || ''}
              onChange={(e) => onCfg({ ...cfg, generator: { ...cfg.generator, system_prompt: e.target.value } })}
              placeholder="System instruction…"
            />
          </Field>
        </div>
      ) : null}

      {selected === 'generator' ? (
        <div className="space-y-3">
          <Field label="HF LLM model">
            <Input value={cfg.generator.model_id} onChange={(e) => onCfg({ ...cfg, generator: { ...cfg.generator, model_id: e.target.value } })} />
          </Field>
          <div className="grid grid-cols-3 gap-2">
            <Field label="Temp">
              <Input
                type="number"
                value={cfg.generator.temperature}
                onChange={(e) => onCfg({ ...cfg, generator: { ...cfg.generator, temperature: Number(e.target.value || 0.1) } })}
              />
            </Field>
            <Field label="Max new">
              <Input
                type="number"
                value={cfg.generator.max_new_tokens}
                onChange={(e) => onCfg({ ...cfg, generator: { ...cfg.generator, max_new_tokens: Math.max(1, Number(e.target.value || 256)) } })}
              />
            </Field>
            <Field label="Top-p">
              <Input
                type="number"
                value={cfg.generator.top_p}
                onChange={(e) => onCfg({ ...cfg, generator: { ...cfg.generator, top_p: Math.max(0.1, Math.min(1, Number(e.target.value || 0.9))) } })}
              />
            </Field>
          </div>
        </div>
      ) : null}

      {selected === 'guardrails' ? (
        <div className="space-y-3">
          <Field label="Require citations">
            <Select
              value={cfg.guardrails?.require_citations ? 'yes' : 'no'}
              onChange={(e) => onCfg({ ...cfg, guardrails: { ...(cfg.guardrails || { min_citations: 1 }), require_citations: e.target.value === 'yes' } })}
            >
              <option value="yes">yes</option>
              <option value="no">no</option>
            </Select>
          </Field>
          <Field label="Min citations">
            <Input
              type="number"
              value={cfg.guardrails?.min_citations ?? 1}
              onChange={(e) => onCfg({ ...cfg, guardrails: { ...(cfg.guardrails || { require_citations: true }), min_citations: Math.max(1, Number(e.target.value || 1)) } })}
            />
          </Field>
        </div>
      ) : null}

      {selected === 'reader' || selected === 'vectorstore' ? (
        <div className="text-sm text-white/60">This module is automatic in local mode.</div>
      ) : null}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-white/55 mb-1">{label}</div>
      {children}
    </div>
  )
}

export function Architect() {
  return (
    <ReactFlowProvider>
      <ArchitectInner />
    </ReactFlowProvider>
  )
}
