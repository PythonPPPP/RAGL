import React from 'react'
import { useApp } from '../store/app'
import { api } from '../api/client'
import type { DatasetItem, DatasetRow, DocumentItem } from '../api/types'
import { Badge, Button, Card, CardContent, CardHeader, Divider, Input, Textarea } from '../components/ui'
import { FileUp, Plus, Sparkles, Trash2 } from 'lucide-react'

function EmptyState({ title, desc, action }: { title: string; desc: string; action?: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/4 p-6">
      <div className="text-lg font-semibold">{title}</div>
      <div className="text-sm text-white/60 mt-1">{desc}</div>
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  )
}

export function DataStudio() {
  const { selectedProject, seedDemo } = useApp()
  const projectId = selectedProject?.id

  const [docs, setDocs] = React.useState<DocumentItem[]>([])
  const [datasets, setDatasets] = React.useState<DatasetItem[]>([])
  const [busy, setBusy] = React.useState(false)
  const [err, setErr] = React.useState<string | null>(null)

  const [dsName, setDsName] = React.useState('Support QA v1')
  const [dsDesc, setDsDesc] = React.useState('')
  const [dsQuestions, setDsQuestions] = React.useState('')

  const fileRef = React.useRef<HTMLInputElement | null>(null)

  async function refresh() {
    if (!projectId) return
    setErr(null)
    const [d, s] = await Promise.all([api.listDocuments(projectId), api.listDatasets(projectId)])
    setDocs(d)
    setDatasets(s)
  }

  React.useEffect(() => {
    refresh().catch((e) => setErr(String(e.message || e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  if (!projectId) {
    return (
      <EmptyState
        title="No project selected"
        desc="Create a project from the top bar to start uploading documents and building datasets."
      />
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-lg font-semibold">Documents</div>
                <div className="text-sm text-white/60">Upload knowledge base files (PDF/DOCX/MD/TXT/HTML/CSV)</div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  ref={fileRef}
                  type="file"
                  className="hidden"
                  onChange={async (e) => {
                    const f = e.target.files?.[0]
                    if (!f) return
                    setBusy(true)
                    setErr(null)
                    try {
                      await api.uploadDocument(projectId, f)
                      await refresh()
                    } catch (ex: any) {
                      setErr(String(ex?.message || ex))
                    } finally {
                      setBusy(false)
                      e.currentTarget.value = ''
                    }
                  }}
                />
                <Button disabled={busy} onClick={() => fileRef.current?.click()}>
                  <FileUp size={16} className="mr-2" />
                  Upload
                </Button>
                <Button
                  variant="secondary"
                  disabled={busy}
                  onClick={async () => {
                    setBusy(true)
                    setErr(null)
                    try {
                      await seedDemo(false)
                      await refresh()
                    } catch (e: any) {
                      setErr(String(e?.message || e))
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  <Sparkles size={16} className="mr-2" />
                  Demo pack
                </Button>
                <Button variant="secondary" disabled={busy} onClick={() => refresh().catch(() => undefined)}>
                  Refresh
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {err ? (
              <div className="mb-4">
                <Badge tone="bad">{err}</Badge>
              </div>
            ) : null}

            {docs.length === 0 ? (
              <div className="space-y-4">
                <div
                  className="rounded-2xl border border-dashed border-white/15 bg-white/3 p-6 text-center"
                  onDragOver={(e) => {
                    e.preventDefault()
                    e.dataTransfer.dropEffect = 'copy'
                  }}
                  onDrop={async (e) => {
                    e.preventDefault()
                    const f = e.dataTransfer.files?.[0]
                    if (!f) return
                    setBusy(true)
                    setErr(null)
                    try {
                      await api.uploadDocument(projectId, f)
                      await refresh()
                    } catch (ex: any) {
                      setErr(String(ex?.message || ex))
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  <div className="text-lg font-semibold">Drop a file here</div>
                  <div className="text-sm text-white/60 mt-1">or click “Upload” to select a document</div>
                  <div className="text-xs text-white/45 mt-3">PDF/DOCX/MD/TXT/HTML/CSV</div>
                </div>

                <EmptyState
                  title="No documents yet"
                  desc="Install the demo pack (one click) or upload your own docs. Then go to Architect to build an index."
                  action={<Badge tone="neutral">Local storage: backend/data</Badge>}
                />
              </div>
            ) : (
              <div className="overflow-hidden rounded-2xl border border-white/10">
                <table className="w-full text-sm">
                  <thead className="bg-white/5 text-white/60">
                    <tr>
                      <th className="text-left px-4 py-3 font-medium">File</th>
                      <th className="text-left px-4 py-3 font-medium">Status</th>
                      <th className="text-left px-4 py-3 font-medium">Meta</th>
                      <th className="text-left px-4 py-3 font-medium">Created</th>
                      <th className="text-right px-4 py-3 font-medium">Actions</th>
                      <th className="text-right px-4 py-3 font-medium">ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {docs.map((d) => (
                      <tr key={d.id} className="border-t border-white/10 hover:bg-white/3">
                        <td className="px-4 py-3">
                          <div className="font-medium">{d.filename}</div>
                          <div className="text-xs text-white/50">project #{d.project_id}</div>
                        </td>
                        <td className="px-4 py-3">
                          <Badge tone={d.status === 'processed' ? 'good' : 'neutral'}>{d.status}</Badge>
                        </td>
                        <td className="px-4 py-3 text-white/60">
                          <div className="text-xs">{d.meta?.size ? `${Math.round((d.meta.size as number) / 1024)} KB` : '—'}</div>
                        </td>
                        <td className="px-4 py-3 text-white/60">
                          <div className="text-xs">{d.created_at ? new Date(d.created_at).toLocaleString() : '—'}</div>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <Button
                            variant="ghost"
                            disabled={busy}
                            onClick={async () => {
                              setBusy(true)
                              setErr(null)
                              try {
                                await api.deleteDocument(d.id)
                                await refresh()
                              } catch (ex: any) {
                                setErr(String(ex?.message || ex))
                              } finally {
                                setBusy(false)
                              }
                            }}
                          >
                            <Trash2 size={16} className="mr-2" />
                            Delete
                          </Button>
                        </td>
                        <td className="px-4 py-3 text-right text-white/60">{d.id}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-lg font-semibold">Quick tips</div>
                <div className="text-sm text-white/60">A solid default workflow for beginners</div>
              </div>
              <Badge tone="neutral">
                <Sparkles size={14} className="mr-1" />
                Local-first
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm text-white/70 leading-relaxed">
              1) Upload 2–10 documents. 2) Go to <span className="text-white">Architect</span> and load a template.
              3) Build index. 4) Test in <span className="text-white">The Lab</span>. 5) Run benchmark and compare in{' '}
              <span className="text-white">Insights</span>.
            </div>
            <Divider />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
                <div className="text-xs text-white/60">Chunk size</div>
                <div className="font-semibold mt-1">800–1100</div>
                <div className="text-xs text-white/50 mt-1">Try 900/120 overlap first</div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
                <div className="text-xs text-white/60">Retriever k</div>
                <div className="font-semibold mt-1">8–20</div>
                <div className="text-xs text-white/50 mt-1">Hybrid + rerank is safer</div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/4 p-4">
                <div className="text-xs text-white/60">LLM</div>
                <div className="font-semibold mt-1">TinyLlama 1.1B</div>
                <div className="text-xs text-white/50 mt-1">Runs on CPU (slow) or GPU</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold">Datasets</div>
                <div className="text-sm text-white/60">Questions for benchmarking</div>
              </div>
              <Badge tone="neutral">{datasets.length} total</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {datasets.length === 0 ? (
              <EmptyState
                title="No datasets"
                desc="Create a simple question list to evaluate pipelines."
              />
            ) : (
              <div className="space-y-2">
                {datasets.map((ds) => (
                  <div key={ds.id} className="rounded-2xl border border-white/10 bg-white/4 p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-semibold">{ds.name}</div>
                        <div className="text-xs text-white/55 mt-1">{ds.description || '—'}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge tone="neutral">{ds.count} q</Badge>
                        <Button
                          variant="ghost"
                          disabled={busy}
                          onClick={async () => {
                            setBusy(true)
                            setErr(null)
                            try {
                              await api.deleteDataset(ds.id)
                              await refresh()
                            } catch (ex: any) {
                              setErr(String(ex?.message || ex))
                            } finally {
                              setBusy(false)
                            }
                          }}
                        >
                          <Trash2 size={16} />
                        </Button>
                      </div>
                    </div>
                    <div className="text-xs text-white/40 mt-2">id: {ds.id}</div>
                  </div>
                ))}
              </div>
            )}

            <Divider />

            <div className="text-sm font-semibold">Create dataset</div>
            <div className="grid grid-cols-1 gap-2">
              <div>
                <div className="text-xs text-white/50 mb-1">Name</div>
                <Input value={dsName} onChange={(e) => setDsName(e.target.value)} placeholder="Support QA v1" />
              </div>
              <div>
                <div className="text-xs text-white/50 mb-1">Description</div>
                <Input value={dsDesc} onChange={(e) => setDsDesc(e.target.value)} placeholder="Optional" />
              </div>
              <div>
                <div className="text-xs text-white/50 mb-1">Questions (one per line)</div>
                <Textarea
                  value={dsQuestions}
                  onChange={(e) => setDsQuestions(e.target.value)}
                  placeholder={`What is the refund policy?\nHow do I reset my password?\n...`}
                />
                <div className="text-xs text-white/40 mt-1">
                  Tip: you can add a reference answer with <span className="text-white/70">question ||| reference</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  disabled={busy}
                  onClick={async () => {
                    const lines = dsQuestions
                      .split(/\r?\n/)
                      .map((l) => l.trim())
                      .filter(Boolean)
                    if (lines.length === 0) {
                      setErr('Add at least one question')
                      return
                    }
                    const data: DatasetRow[] = lines.map((l) => {
                      const parts = l.split('|||').map((p) => p.trim())
                      if (parts.length >= 2) return { question: parts[0], reference: parts.slice(1).join(' ||| ') }
                      return { question: l }
                    })
                    setBusy(true)
                    setErr(null)
                    try {
                      await api.createDataset(projectId, dsName.trim() || 'Dataset', dsDesc, data)
                      setDsQuestions('')
                      await refresh()
                    } catch (ex: any) {
                      setErr(String(ex?.message || ex))
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  <Plus size={16} className="mr-2" />
                  Create
                </Button>
                <Badge tone="neutral">Stored in SQLite</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
