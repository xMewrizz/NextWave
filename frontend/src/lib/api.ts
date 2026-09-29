// Единственное место с типами контракта backend. Меняется контракт — меняется этот файл;
// backend/test_frontend_contract.py сверяет типы с backend/app/models.py и demo/result.

export type JobStatus = 'pending' | 'running' | 'complete' | 'error'
export type JobMode = 'cached_snapshot' | 'live'
export type StageStatus = 'pending' | 'running' | 'complete' | 'reused' | 'error'
export type Bucket = 'main' | 'watchlist' | 'excluded'

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

export interface Health {
  status: 'ok'
  mode: JobMode
}

export interface ModelFactor {
  feature_name: string
  label_ru: string
  feature_group: string
  raw_value: string | number | boolean
  transformed_value: number
  coefficient: number
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
  automatic_translation: boolean
  generated_summary: boolean
}

export interface EvidenceClaimView {
  claim_id: string
  kind: string
  direction: string
  scope: string
  quote: string
  explanation_ru: string
  interpretation_generated: boolean
  source: EvidenceSource
}

export interface ResultModel {
  score: number
  threshold: number
  prediction: number
  score_semantics: string
  confidence: null
  top_positive_factors: ModelFactor[]
  top_negative_factors: ModelFactor[]
}

export interface EvidenceReview {
  status: string
  full_candidate_claims: number
  support_claims: number
  counter_claims: number
  independent_origins: number
  independent_actors: number
  grounded_ab_support: boolean
}

export interface ResultCandidate {
  schema_version: string
  candidate_id: string
  canonical_name: string
  aliases: string[]
  domain: string
  source_query: string
  status: Bucket
  reason: string
  reason_ru: string
  top15_rank?: number
  duplicate_of?: string
  duplicate_of_name?: string
  duplicate_candidate_ids?: string[]
  description_ru: string | null
  potential_advantage_ru: string | null
  case_example: EvidenceClaimView | null
  model: ResultModel
  evidence_review: EvidenceReview
  signal_case: EvidenceClaimView[]
  skeptic_case: EvidenceClaimView[]
  limitations: string[]
}

export interface ResultSummary {
  schema_version: string
  source_query: string
  cutoff_date: string
  release_status: string
  analysis_status: string
  main_target: number
  main_target_met: boolean
  candidate_count: number
  top15_count: number
  processed_document_relations: number
  processed_unique_documents: number
  processed_unique_origins: number
  candidates_model_score_gt_075: number
  main_model_score_gt_075: number
  high_confidence_weak_signals: null
  confidence_available: false
  confidence_note: string
  status_counts: Partial<Record<Bucket, number>>
  reason_counts: Record<string, number>
  candidate_gate?: { evaluated_proposals: number }
}

export interface ResultQuery {
  text: string
  cutoff_date: string
}

export interface ResultBundle {
  schema_version: 'analysis-response-v1'
  query: ResultQuery
  status: string
  main_target: number
  summary: ResultSummary
  top15: ResultCandidate[]
  candidates: ResultCandidate[]
}

export const JOB_TERMINAL: JobStatus[] = ['complete', 'error']

/** Ошибка API с уже готовым русским текстом; status = 0 — сервер не ответил. */
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

/** FastAPI отдаёт detail строкой (HTTPException) или списком (422 валидации pydantic). */
function errorMessage(status: number, detail: unknown): string {
  if (status === 422 && typeof detail !== 'string') {
    return 'Запрос не принят: он должен быть непустым и не длиннее 200 символов.'
  }
  if (typeof detail === 'string' && detail) {
    return status === 500 ? `Внутренняя ошибка сервиса: ${detail}` : detail
  }
  if (status === 502 || status === 503 || status === 504) {
    return `Сервис временно недоступен (${status}). Попробуйте позже.`
  }
  return `Сервис ответил ошибкой (${status}).`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { 'content-type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ApiError('Сервер не отвечает. Проверьте, что backend запущен, и повторите попытку.', 0)
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ApiError(errorMessage(response.status, body?.detail), response.status)
  }
  return response.json()
}

function checkJob(job: AnalysisJob): AnalysisJob {
  if (job.schema_version !== 'analysis-job-v1') {
    throw new ApiError('Сервер вернул анализ неподдерживаемой версии.', 0)
  }
  return job
}

function checkResult(result: ResultBundle): ResultBundle {
  if (
    result.schema_version !== 'analysis-response-v1' ||
    !Array.isArray(result.top15) ||
    !Array.isArray(result.candidates)
  ) {
    throw new ApiError('Сервер вернул результат неподдерживаемой версии.', 0)
  }
  return result
}

export const api = {
  health: () => request<Health>('/health'),
  history: () => request<AnalysisJob[]>('/analyses').then((jobs) => jobs.map(checkJob)),
  analysisJob: (id: string) => request<AnalysisJob>(`/analyses/${id}`).then(checkJob),
  analysisResult: (id: string) => request<ResultBundle>(`/analyses/${id}/result`).then(checkResult),
  startAnalysis: (query: string) =>
    request<AnalysisJob>('/analyses', { method: 'POST', body: JSON.stringify({ query }) }).then(checkJob),
  retryAnalysis: (id: string) =>
    request<AnalysisJob>(`/analyses/${id}/retry`, { method: 'POST' }).then(checkJob),
  currentResult: () => request<ResultBundle>('/result/current').then(checkResult),
}
