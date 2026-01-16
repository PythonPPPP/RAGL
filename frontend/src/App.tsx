import { Navigate, Route, Routes } from 'react-router-dom'
import { Shell } from './components/Shell'
import { DataStudio } from './pages/DataStudio'
import { Architect } from './pages/Architect'
import { TheLab } from './pages/TheLab'
import { Insights } from './pages/Insights'

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/data" element={<DataStudio />} />
        <Route path="/architect" element={<Architect />} />
        <Route path="/lab" element={<TheLab />} />
        <Route path="/insights" element={<Insights />} />
        <Route path="/" element={<Navigate to="/data" replace />} />
      </Routes>
    </Shell>
  )
}
