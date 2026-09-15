import type {
  Analysis,
  AnalysisStatus,
  AnalysisSummary,
  CandidateStatus,
  Coverage,
  FactorKey,
  SourceType,
  Stage,
  Trend,
} from '@/lib/contracts.generated'

export type {
  Analysis,
  AnalysisStatus,
  AnalysisSummary,
  CandidateAssessment,
  CandidateFeatures,
  CandidateStatus,
  Coverage,
  DevelopmentStage,
  Evidence,
  ExclusionReason,
  FactorKey,
  ModelPrediction,
  ScoreFactor,
  SourceStat,
  SourceType,
  Stage,
  TimelinePoint,
  Trend,
  TrustLevel,
  UseCase,
} from '@/lib/contracts.generated'

export type Bucket = CandidateStatus

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

/** Canonical ranking score normalized to the 0..1 scale used by the UI. */
export function trendScore(trend: Trend): number {
  return trend.priority_score === null
    ? trend.prediction.weak_signal_score
    : trend.priority_score / 100
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
  scientific_publication: 'научная публикация',
  patent: 'патент',
  standard: 'стандарт',
  regulator: 'регулятор',
  university: 'университет',
  company: 'компания',
  industry_media: 'отраслевое медиа',
  analytical_report: 'аналитический отчёт',
  conference: 'конференция',
  social_or_blog: 'блог или соцсеть',
  other: 'другое',
}
