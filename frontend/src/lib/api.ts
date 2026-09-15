// Типы повторяют backend/app/models.py. Меняется контракт — меняются оба файла.

export type AnalysisStatus = 'pending' | 'running' | 'done' | 'empty' | 'error'
export type SourceType = 'preprint' | 'journal' | 'patent' | 'vendor' | 'conference' | 'report'
export type FactorKey = 'growth' | 'novelty' | 'independence' | 'evidence'
export type Bucket = 'main' | 'watchlist' | 'excluded'

export interface SourceRef {
  title: string
  url: string
  source_type: SourceType
  published_at: string | null
}

export interface TimelinePoint {
  period: string
  documents: number
  share: number
}

export interface ScoreFactor {
  key: FactorKey
  value: number
  explanation: string
}

export interface UseCase {
  title: string
  organization: string | null
  description: string
  url: string
}

export interface Trend {
  id: string
  rank: number
  bucket: Bucket
  bucket_reason: string
  title: string
  summary: string
  score: number
  factors: ScoreFactor[]
  problem: string
  advantage: string
  hypothesis: string | null
  use_case: UseCase
  first_seen: string
  timeline: TimelinePoint[]
  sources: SourceRef[]
  document_count: number
  independent_sources: number
  limitations: string[]
}

export interface SourceStat {
  name: string
  source_type: SourceType
  documents: number
}

export interface Coverage {
  directions: string[]
  examples: string[]
  documents_from: string
  documents_to: string
  document_count: number
  sources: SourceStat[]
  corpus_version: string
  method_version: string
  updated_at: string
  thresholds: Record<string, number>
}

export interface Stage {
  key: string
  label: string
}

export interface Analysis {
  id: string
  query: string
  status: AnalysisStatus
  stage: string | null
  progress: number
  notice: string | null
  created_at: string
  finished_at: string | null
  corpus_version: string
  method_version: string
  trends: Trend[]
}

export interface AnalysisSummary {
  id: string
  query: string
  status: AnalysisStatus
  created_at: string
  trend_count: number
}

export const TERMINAL: AnalysisStatus[] = ['done', 'empty', 'error']

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { 'content-type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail ?? `Сервис недоступен (${response.status})`)
  }
  return response.json()
}

export const api = {
  coverage: () => request<Coverage>('/coverage'),
  stages: () => request<Stage[]>('/stages'),
  history: () => request<AnalysisSummary[]>('/analyses'),
  analysis: (id: string) => request<Analysis>(`/analyses/${id}`),
  startAnalysis: (query: string) =>
    request<Analysis>('/analyses', { method: 'POST', body: JSON.stringify({ query }) }),
}

export const FACTOR_LABELS: Record<FactorKey, string> = {
  growth: 'Рост',
  novelty: 'Новизна',
  independence: 'Независимость',
  evidence: 'Доказательная база',
}

export const BUCKETS: {
  key: Bucket
  label: string
  short: string
  description: string
  accent: string
  dot: string
}[] = [
  {
    key: 'main',
    label: 'Зарождающиеся тренды',
    short: 'Основной список',
    description:
      'Темы с признаками нового развития, прошедшие пороги по новизне, росту и доказательной базе. Не более 15 кандидатов, отсортированы по рейтингу.',
    accent: 'text-foreground',
    dot: 'bg-foreground',
  },
  {
    key: 'watchlist',
    label: 'Наблюдение',
    short: 'Наблюдение',
    description:
      'Признаки зарождения есть, но подтверждений пока мало: единичные публикации, зависимые источники или выводы на уровне предположений. Стоит проверить при следующем обновлении корпуса.',
    accent: 'text-muted-foreground',
    dot: 'bg-muted-foreground',
  },
  {
    key: 'excluded',
    label: 'Отсеяны',
    short: 'Отсеяны',
    description:
      'Темы, не прошедшие проверку на новизну или зарождаемость: устоявшиеся направления и темы, доля которых в корпусе перестала расти. Показаны, чтобы отбор можно было проверить.',
    accent: 'text-muted-foreground',
    dot: 'border border-foreground bg-transparent',
  },
]

export const SOURCE_LABELS: Record<SourceType, string> = {
  preprint: 'препринт',
  journal: 'журнал',
  patent: 'патент',
  vendor: 'вендор',
  conference: 'конференция',
  report: 'отчёт',
}
