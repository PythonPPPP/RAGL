import type {
  ChatAnswer,
  DatasetItem,
  DatasetRow,
  PipelineConfig,
  PipelineDetail,
  PipelineItem,
  Project,
  DocumentItem,
  RunDetail,
  RunItem,
  BuildJobStatus,
  ShareCreateResult,
  SharePayload,
  RetrieveDebug,
  GenerateResult,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.headers || {}),
    },
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const j = await res.json()
      detail = j.detail || JSON.stringify(j)
    } catch {
      try {
        detail = await res.text()
      } catch {
        // ignore
      }
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export const api = {
  status: () => apiFetch<{ engine: string; ready: boolean; message: string }>('/status'),

  // demo
  seedDemo: (force = false, build_index = true) =>
    apiFetch<{ ok: boolean; project_id: number; message: string; pipelines?: Record<string, number>; index_built_for?: number[] }>(
      '/demo/seed',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force, build_index }),
      }
    ),

  // projects
  listProjects: () => apiFetch<Project[]>('/projects'),
  createProject: (name: string, description: string) =>
    apiFetch<Project>('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    }),

  // documents
  listDocuments: (project_id: number) => apiFetch<DocumentItem[]>(`/documents?project_id=${project_id}`),
  deleteDocument: (doc_id: number) => apiFetch<{ ok: boolean }>(`/documents/${doc_id}`, { method: 'DELETE' }),
  uploadDocument: async (project_id: number, file: File) => {
    const form = new FormData()
    form.append('project_id', String(project_id))
    form.append('file', file)
    const res = await fetch(`${API_BASE}/documents/upload`, { method: 'POST', body: form })
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text)
    }
    return (await res.json()) as DocumentItem
  },

  // datasets
  listDatasets: (project_id: number) => apiFetch<DatasetItem[]>(`/datasets?project_id=${project_id}`),
  getDataset: (dataset_id: number) => apiFetch<{ id: number; project_id: number; name: string; description: string; data: DatasetRow[] }>(`/datasets/${dataset_id}`),
  createDataset: (project_id: number, name: string, description: string, data: DatasetRow[]) =>
    apiFetch<DatasetItem>('/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id, name, description, data }),
    }),
  deleteDataset: (dataset_id: number) => apiFetch<{ ok: boolean }>(`/datasets/${dataset_id}`, { method: 'DELETE' }),

  // pipelines
  templates: () => apiFetch<any>('/pipelines/templates'),
  listPipelines: (project_id: number) => apiFetch<PipelineItem[]>(`/pipelines?project_id=${project_id}`),
  getPipeline: (pipeline_id: number) => apiFetch<PipelineDetail>(`/pipelines/${pipeline_id}`),
  createPipeline: (project_id: number, name: string, description: string, config: PipelineConfig) =>
    apiFetch<PipelineItem>('/pipelines', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id, name, description, config }),
    }),
  updatePipeline: (pipeline_id: number, project_id: number, name: string, description: string, config: PipelineConfig) =>
    apiFetch<{ ok: boolean }>(`/pipelines/${pipeline_id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id, name, description, config }),
    }),
  deletePipeline: (pipeline_id: number, delete_index = false) =>
    apiFetch<{ ok: boolean }>(`/pipelines/${pipeline_id}?delete_index=${delete_index ? 'true' : 'false'}`, { method: 'DELETE' }),
  duplicatePipeline: (pipeline_id: number, name?: string, description?: string) =>
    apiFetch<PipelineItem>(`/pipelines/${pipeline_id}/duplicate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    }),
  buildIndex: (pipeline_id: number) => apiFetch<any>(`/pipelines/${pipeline_id}/build_index`, { method: 'POST' }),
  buildIndexAsync: (pipeline_id: number) => apiFetch<{ ok: boolean; job_id: string; status: string }>(`/pipelines/${pipeline_id}/build_index_async`, { method: 'POST' }),
  buildIndexStatus: (pipeline_id: number) => apiFetch<BuildJobStatus>(`/pipelines/${pipeline_id}/build_index_status`),

  // share
  createShare: (name: string, description: string, config: PipelineConfig) =>
    apiFetch<ShareCreateResult>('/share', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description, config }),
    }),
  getShare: (code: string) => apiFetch<SharePayload>(`/share/${encodeURIComponent(code)}`),

  // playground
  retrieveDebug: (project_id: number, pipeline_id: number, question: string, override_model_id?: string) =>
    apiFetch<RetrieveDebug>('/playground/retrieve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id, pipeline_id, question, override_model_id }),
    }),
  generate: (model_id: string, prompt: string, temperature: number, max_new_tokens: number, top_p: number) =>
    apiFetch<GenerateResult>('/playground/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id, prompt, temperature, max_new_tokens, top_p }),
    }),

  // chat
  ask: (project_id: number, pipeline_id: number, question: string, override_model_id?: string) =>
    apiFetch<ChatAnswer>('/chat/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id, pipeline_id, question, override_model_id }),
    }),

  // eval & runs
  startEval: (project_id: number, pipeline_id: number, dataset_id: number, notes: string, async_run = true) =>
    apiFetch<{ run_id: number; status: string; job_id: string }>('/eval/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id, pipeline_id, dataset_id, notes, async_run }),
    }),
  listRuns: (project_id: number) => apiFetch<RunItem[]>(`/runs?project_id=${project_id}`),
  getRun: (run_id: number) => apiFetch<RunDetail>(`/runs/${run_id}`),
}
