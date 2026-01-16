export type Project = {
  id: number
  name: string
  description: string
}

export type DocumentItem = {
  id: number
  project_id: number
  filename: string
  status: string
  meta?: Record<string, any>
  created_at?: string
}

export type DatasetItem = {
  id: number
  project_id: number
  name: string
  description: string
  count: number
}

export type DatasetRow = {
  question: string
  reference?: string
}

export type RetrieverConfig = {
  mode: 'dense' | 'bm25' | 'hybrid'
  top_k: number
  hybrid_alpha?: number
  use_reranker: boolean
  rerank_top_n: number
  min_score?: number | null
  use_mmr?: boolean
  mmr_lambda?: number
  mmr_k?: number
}

export type QueryConfig = {
  mode: 'none' | 'keywords' | 'multi'
  max_queries: number
}

export type ContextConfig = {
  max_chars: number
}

export type GuardrailsConfig = {
  require_citations: boolean
  min_citations: number
}

export type ChunkerConfig = {
  chunk_size: number
  overlap: number
}

export type GeneratorConfig = {
  model_id: string
  temperature: number
  max_new_tokens: number
  top_p: number
  system_prompt?: string
}

export type PipelineConfig = {
  name?: string
  description?: string
  chunker: ChunkerConfig
  embedding_model: string
  reranker_model: string
  query?: QueryConfig
  context?: ContextConfig
  guardrails?: GuardrailsConfig
  retriever: RetrieverConfig
  generator: GeneratorConfig
  ui?: any
}

export type PipelineItem = {
  id: number
  project_id: number
  name: string
  description: string
}

export type PipelineDetail = {
  id: number
  project_id: number
  name: string
  description: string
  config: PipelineConfig
  index_ready: boolean
}

export type ChatAnswer = {
  answer: string
  sources: Array<{
    rank: number
    score: number
    doc_id: number
    filename: string
    // Backend uses a string id ("<doc_id>:<chunk_index>")
    chunk_id: string
    text: string
  }>
  timings: Record<string, number>
  tokens?: { input: number; output: number; total?: number }
  cost_usd?: number
  prompt_preview?: string
  context_chars?: number
  pipeline: string
}

export type BuildJobStatus = {
  job_id: string
  status: 'unknown' | 'queued' | 'running' | 'done' | 'error'
  result?: any
  error?: string | null
}

export type ShareCreateResult = {
  ok: boolean
  code: string
}

export type SharePayload = {
  ok: boolean
  code: string
  name: string
  description: string
  config: PipelineConfig
  created_at: string
}

export type RetrieveDebug = {
  ok: boolean
  queries: string[]
  context_preview: string
  context_chars: number
  sources: ChatAnswer['sources']
  timings: Record<string, number>
}

export type GenerateResult = {
  ok: boolean
  text: string
  tokens: { input: number; output: number }
  cost_usd: number
}

export type RunItem = {
  id: number
  project_id: number
  pipeline_id: number
  dataset_id?: number
  status: string
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  metrics: Record<string, any>
  notes: string
}

export type RunDetail = {
  id: number
  project_id: number
  pipeline_id: number
  dataset_id?: number
  status: string
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  notes: string
  metrics: Record<string, any>
  report?: any
  artifacts_dir?: string | null
}
