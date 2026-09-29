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

export type JobStatus = 'pending' | 'running' | 'complete' | 'error'
export type JobMode = 'cached_snapshot' | 'live'
export type StageStatus = 'pending' | 'running' | 'complete' | 'reused' | 'error'

export interface AnalysisStageState {
  key: string
  label: string
  status: StageStatus
}

export interface AnalysisJob {
  schema_version: 'analysis-job-v1'
  id: string
  query: string
  mode: JobMode
  status: JobStatus
  stage: string | null
  stage_label: string | null
  progress: number
  created_at: string
  updated_at: string
  finished_at: string | null
  result_available: boolean
  result_sha256: string | null
  stage_history: AnalysisStageState[]
  error: string | null
}

export interface ModelFactor {
  feature_name: string
  label_ru: string
  feature_group: string
  raw_value: string | number | boolean
  contribution: number
  direction: string
}

export interface EvidenceSource {
  document_id: string
  name: string | null
  url: string | null
  published_at: string | null
  source_type: string | null
  language: string | null
  trust_tier: string
  publisher: string | null
}

export interface EvidenceClaimView {
  claim_id: string
  kind: string
  direction: string
  quote: string
  explanation_ru: string
  source: EvidenceSource
}

export interface ResultCandidate {
  candidate_id: string
  canonical_name: string
  aliases: string[]
  domain: string
  source_query: string
  status: Bucket
  reason: string
  reason_ru: string
  top15_rank?: number
  description_ru: string | null
  potential_advantage_ru: string | null
  model: {
    score: number
    threshold: number
    prediction: number
    score_semantics: string
    confidence: null
    top_positive_factors: ModelFactor[]
    top_negative_factors: ModelFactor[]
  }
  evidence_review: {
    status: string
    full_candidate_claims: number
    support_claims: number
    counter_claims: number
    independent_origins: number
    independent_actors: number
    grounded_ab_support: boolean
  }
  signal_case: EvidenceClaimView[]
  skeptic_case: EvidenceClaimView[]
  limitations: string[]
}

export interface ResultSummary {
  source_query: string
  release_status: string
  candidate_count: number
  top15_count: number
  processed_document_relations: number
  processed_unique_documents: number
  processed_unique_origins: number
  candidates_model_score_gt_075: number
  high_confidence_weak_signals: null
  confidence_available: false
  status_counts: Record<Bucket, number>
  reason_counts: Record<string, number>
  candidate_gate?: {
    evaluated_proposals: number
  }
}

export interface ResultBundle {
  schema_version: 'analysis-response-v1'
  query: { text: string; cutoff_date: string }
  status: string
  main_target: number
  summary: ResultSummary
  top15: ResultCandidate[]
  candidates: ResultCandidate[]
}

export const TERMINAL: AnalysisStatus[] = ['done', 'empty', 'error']
export const JOB_TERMINAL: JobStatus[] = ['complete', 'error']

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
  history: () => request<AnalysisJob[]>('/analyses'),
  analysis: (id: string) => request<Analysis>(`/analyses/${id}`),
  analysisJob: (id: string) => request<AnalysisJob>(`/analyses/${id}`),
  analysisResult: (id: string) => request<ResultBundle>(`/analyses/${id}/result`),
  startAnalysis: (query: string) =>
    request<AnalysisJob>('/analyses', { method: 'POST', body: JSON.stringify({ query }) }),
  currentResult: () => request<ResultBundle>('/result/current'),
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
