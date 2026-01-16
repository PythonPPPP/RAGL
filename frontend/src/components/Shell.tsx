import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { AppProvider, useApp } from '../store/app'
import { Badge, Button, Card, CardContent, Input, Select, cn } from './ui'
import { LayoutDashboard, Database, FlaskConical, Network, BarChart3, Settings2 } from 'lucide-react'

function TopNav() {
  const loc = useLocation()
  const { status, projects, selectedProject, setSelectedProjectId, createProject, seedDemo } = useApp()
  const [showNew, setShowNew] = React.useState(false)
  const [showSettings, setShowSettings] = React.useState(false)
  const [newName, setNewName] = React.useState('My Project')
  const [newDesc, setNewDesc] = React.useState('')
  const [settingsMsg, setSettingsMsg] = React.useState<string | null>(null)
  const [settingsBusy, setSettingsBusy] = React.useState(false)

  const tabs = [
    { href: '/data', label: 'Data Studio', icon: Database },
    { href: '/architect', label: 'Architect', icon: Network },
    { href: '/lab', label: 'The Lab', icon: FlaskConical },
    { href: '/insights', label: 'Insights', icon: BarChart3 },
  ]

  return (
    <div className="sticky top-0 z-50 border-b border-white/10 bg-bg-900/60 backdrop-blur">
      <div className="mx-auto max-w-[1440px] px-5 py-3 flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-sky-500/90 grid place-items-center font-bold">R</div>
          <div>
            <div className="text-sm font-semibold">RAGL <span className="text-white/40 font-medium">v0.9</span></div>
            <div className="text-xs text-white/50 -mt-0.5">RAG like Lego</div>
          </div>
        </div>

        <div className="ml-2 flex-1 flex justify-center">
          <div className="inline-flex rounded-2xl bg-white/5 border border-border-700 p-1">
            {tabs.map(t => {
              const active = loc.pathname.startsWith(t.href)
              const Icon = t.icon
              return (
                <Link
                  key={t.href}
                  to={t.href}
                  className={cn(
                    'px-4 h-9 rounded-xl flex items-center gap-2 text-sm transition',
                    active ? 'bg-white/10 text-white' : 'text-white/60 hover:text-white hover:bg-white/5'
                  )}
                >
                  <Icon size={16} />
                  {t.label}
                </Link>
              )
            })}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2">
            <div className="text-xs text-white/50">Project</div>
            <Select
              value={selectedProject?.id ?? ''}
              onChange={e => setSelectedProjectId(Number(e.target.value))}
              className="w-[220px]"
            >
              {projects.length === 0 ? (
                <option value="">—</option>
              ) : (
                projects.map(p => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))
              )}
            </Select>
            <Button variant="secondary" onClick={() => setShowNew(true)}>
              New
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <div className={cn('h-2 w-2 rounded-full', status?.ready ? 'bg-emerald-400' : 'bg-rose-400')} />
            <span className="text-xs text-white/70">{status?.message ?? 'Checking…'}</span>
          </div>

          <Button variant="ghost" className="hidden sm:inline-flex" onClick={() => setShowSettings(true)}>
            <Settings2 size={16} />
          </Button>
        </div>
      </div>

      {showSettings ? (
        <div className="fixed inset-0 z-[100]">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowSettings(false)} />
          <div className="absolute right-0 top-0 h-full w-[420px] max-w-[90vw] border-l border-white/10 bg-bg-900/95 backdrop-blur">
            <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold">Settings</div>
                <div className="text-xs text-white/50">Local engine · demo tools</div>
              </div>
              <Button variant="ghost" onClick={() => setShowSettings(false)}>
                Close
              </Button>
            </div>
            <div className="p-5 space-y-4">
              <Card className="bg-white/4">
                <CardContent className="py-4 space-y-3">
                  <div className="text-sm font-semibold flex items-center justify-between">
                    <span>Demo pack</span>
                    <Badge tone="neutral">recommended</Badge>
                  </div>
                  <div className="text-xs text-white/60 leading-relaxed">
                    Creates a ready-to-try project with documents, datasets and pipelines. The index is built for the
                    default pipeline so you can go straight to <span className="text-white/80">The Lab</span>.
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      disabled={settingsBusy}
                      onClick={async () => {
                        setSettingsBusy(true)
                        setSettingsMsg(null)
                        try {
                          await seedDemo(false)
                          setSettingsMsg('Demo pack installed. Switched to Demo Project.')
                        } catch (e: any) {
                          setSettingsMsg(String(e?.message || e))
                        } finally {
                          setSettingsBusy(false)
                        }
                      }}
                    >
                      Install demo
                    </Button>
                    <Button
                      variant="secondary"
                      disabled={settingsBusy}
                      onClick={async () => {
                        setSettingsBusy(true)
                        setSettingsMsg(null)
                        try {
                          await seedDemo(true)
                          setSettingsMsg('Demo pack reset and re-created.')
                        } catch (e: any) {
                          setSettingsMsg(String(e?.message || e))
                        } finally {
                          setSettingsBusy(false)
                        }
                      }}
                    >
                      Reset demo
                    </Button>
                  </div>
                  {settingsMsg ? <Badge tone={settingsMsg.toLowerCase().includes('error') ? 'bad' : 'neutral'}>{settingsMsg}</Badge> : null}
                </CardContent>
              </Card>

              <Card className="bg-white/4">
                <CardContent className="py-4 space-y-2">
                  <div className="text-sm font-semibold">Quick start</div>
                  <div className="text-xs text-white/60 leading-relaxed">
                    1) Install demo pack (above) or upload your docs in <span className="text-white/80">Data Studio</span>.
                    2) In <span className="text-white/80">Architect</span> pick a template and build index.
                    3) Test in <span className="text-white/80">The Lab</span>. 4) Benchmark → <span className="text-white/80">Insights</span>.
                  </div>
                </CardContent>
              </Card>

              <div className="text-xs text-white/50">
                Tip: If your machine is slow, switch to a smaller HF model in Architect → LLM Generator.
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {showNew && (
        <div className="border-t border-white/10 bg-bg-900/80">
          <div className="mx-auto max-w-[1440px] px-5 py-4">
            <Card className="max-w-[720px]">
              <div className="px-5 py-4 border-b border-border-700 flex items-center justify-between">
                <div className="font-semibold">Create project</div>
                <Button variant="ghost" onClick={() => setShowNew(false)}>Close</Button>
              </div>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <div className="text-xs text-white/50 mb-1">Name</div>
                    <Input value={newName} onChange={e => setNewName(e.target.value)} placeholder="My Project" />
                  </div>
                  <div>
                    <div className="text-xs text-white/50 mb-1">Description</div>
                    <Input value={newDesc} onChange={e => setNewDesc(e.target.value)} placeholder="Optional" />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    onClick={async () => {
                      await createProject(newName.trim() || 'My Project', newDesc)
                      setShowNew(false)
                    }}
                  >
                    Create
                  </Button>
                  <Badge tone="neutral">Stored locally (SQLite + files)</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}

export function Shell({ children }: { children: React.ReactNode }) {
  return (
    <AppProvider>
      <div className="min-h-screen">
        <TopNav />
        <main className="mx-auto max-w-[1440px] px-5 py-6">
          {children}
        </main>
      </div>
    </AppProvider>
  )
}
