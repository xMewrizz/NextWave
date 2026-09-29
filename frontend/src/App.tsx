import { Route, Routes } from 'react-router-dom'
import { AppShell } from '@/components/app-shell'
import { MethodologyPage } from '@/pages/methodology-page'
import { ResultsPage } from '@/pages/results-page'
import { CandidateReportPage, RealResultPage } from '@/pages/real-result-page'
import { SearchPage } from '@/pages/search-page'
import { TrendPage } from '@/pages/trend-page'

export default function App() {
  return (
    <AppShell>
      <div className="min-h-full bg-background">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/analyses/:analysisId" element={<ResultsPage />} />
          <Route path="/result" element={<RealResultPage />} />
          <Route path="/result/candidates/:candidateId" element={<CandidateReportPage />} />
          <Route path="/analyses/:analysisId/candidates/:candidateId" element={<CandidateReportPage />} />
          <Route path="/analyses/:analysisId/trends/:trendId" element={<TrendPage />} />
          <Route path="/methodology" element={<MethodologyPage />} />
        </Routes>
      </div>
    </AppShell>
  )
}
