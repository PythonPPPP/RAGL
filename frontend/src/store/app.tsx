import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import type { Project } from '../api/types'
import { api } from '../api/client'

type AppState = {
  status: { ready: boolean; message: string } | null
  projects: Project[]
  selectedProject: Project | null
  setSelectedProjectId: (id: number) => void
  refreshProjects: () => Promise<void>
  createProject: (name: string, description: string) => Promise<Project>
  seedDemo: (force?: boolean) => Promise<number>
}

const Ctx = createContext<AppState | null>(null)

export function useApp() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp must be used inside AppProvider')
  return v
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<AppState['status']>(null)
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(() => {
    try {
      const raw = localStorage.getItem('rag_arch_project_id')
      if (!raw) return null
      const v = Number(raw)
      return Number.isFinite(v) ? v : null
    } catch {
      return null
    }
  })

  async function refreshProjects() {
    const ps = await api.listProjects()
    setProjects(ps)
    if (ps.length > 0 && (selectedProjectId === null || !ps.find(p => p.id === selectedProjectId))) {
      setSelectedProjectId(ps[0].id)
    }
  }

  useEffect(() => {
    try {
      if (selectedProjectId) localStorage.setItem('rag_arch_project_id', String(selectedProjectId))
    } catch {
      // ignore
    }
  }, [selectedProjectId])

  async function createProject(name: string, description: string) {
    const p = await api.createProject(name, description)
    await refreshProjects()
    setSelectedProjectId(p.id)
    return p
  }

  async function seedDemo(force = false) {
    const res = await api.seedDemo(force, true)
    await refreshProjects()
    setSelectedProjectId(res.project_id)
    return res.project_id
  }

  useEffect(() => {
    api.status().then(s => setStatus({ ready: s.ready, message: s.message })).catch(() => setStatus({ ready: false, message: 'API unreachable' }))
    refreshProjects().catch(() => {
      setProjects([])
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const selectedProject = useMemo(() => projects.find(p => p.id === selectedProjectId) || null, [projects, selectedProjectId])

  const value: AppState = {
    status,
    projects,
    selectedProject,
    setSelectedProjectId: (id: number) => setSelectedProjectId(id),
    refreshProjects,
    createProject,
    seedDemo,
  }

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
