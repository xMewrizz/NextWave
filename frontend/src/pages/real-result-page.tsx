import { useMemo, useState } from 'react'
import { ArrowLeft, ExternalLink, FileText, Info, Minus, Plus } from 'lucide-react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { api, type Bucket, type EvidenceClaimView, type ResultBundle, type ResultCandidate } from '@/lib/api'
import { useResource } from '@/lib/hooks'

const labels: Record<Bucket, string> = {
  main: 'Основная выдача',
  watchlist: 'Требуют наблюдения',
  excluded: 'Исключены',
}

const reasonDescriptions: Record<string, string> = {
  duplicate: 'Объединено с другой карточкой той же технологии.',
  evidence_review_incomplete: 'Проверка доказательств не завершена; тему следует наблюдать.',
  mature: 'Технология уже широко сформировалась и не относится к слабым сигналам.',
  marketing_hype: 'Найдены в основном рекламные утверждения без достаточного независимого подтверждения.',
  temporal_coverage_incomplete: 'Для сравнения динамики не хватает документов в одном из временных окон.',
  insufficient_origins: 'Пока недостаточно независимых первоисточников для основной выдачи.',
  insufficient_actors: 'Подтверждения связаны с недостаточным числом независимых организаций или издателей.',
  insufficient_trusted_evidence: 'Не хватает подтверждений из источников с высоким уровнем доверия.',
  model_below_threshold: 'Набор наблюдаемых признаков пока недостаточно похож на слабый сигнал.',
  passed: 'Тема получила признаки слабого сигнала и подтверждена несколькими независимыми источниками.',
}

export function RealResultPage() {
  const { data, error, loading } = useResource(api.currentResult, 'current-result-page')
  if (loading) return <Message>Загружаю проверенный результат…</Message>
  if (error || !data) return <Message error={error ?? 'Результат отсутствует'} />
  return <ResultView data={data} />
}

export function ResultView({ data }: { data: ResultBundle }) {
  const location = useLocation()
  const [view, setView] = useState<'top15' | 'watchlist' | 'excluded'>('top15')
  const candidates = useMemo(
    () => view === 'top15'
      ? [...data.top15].sort(compareCandidates)
      : data.candidates
          .filter((candidate) => candidate.status === view)
          .sort(compareCandidates),
    [data, view],
  )
  const summary = data.summary
  const reportBase = location.pathname.startsWith('/analyses/')
    ? location.pathname.replace(/\/candidates\/.*$/u, '')
    : '/result'

  const views = [
    { key: 'top15' as const, label: 'TOP-15', count: summary.top15_count },
    { key: 'watchlist' as const, label: 'Наблюдение', count: summary.status_counts.watchlist ?? 0 },
    { key: 'excluded' as const, label: 'Исключены', count: summary.status_counts.excluded ?? 0 },
  ]

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-8 sm:py-12">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
            Результат анализа открытых источников
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-balance">
            {summary.source_query}
          </h1>
          <div className="mt-4 flex flex-wrap gap-2" aria-label="Разделы результата">
            {views.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setView(item.key)}
                className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${view === item.key ? 'bg-foreground text-background' : 'bg-card hover:bg-muted'}`}
              >
                {item.label}
                <span className={`tabular-nums ${view === item.key ? 'text-background/70' : 'text-muted-foreground'}`}>{item.count}</span>
              </button>
            ))}
          </div>
        </div>
        <Badge variant="secondary">Анализ завершён</Badge>
      </div>

      <section className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Названий проверено" value={summary.candidate_gate?.evaluated_proposals ?? 0} />
        <Stat label="Тем после объединения" value={summary.candidate_count} />
        <Stat label="В итоговом TOP-15" value={summary.top15_count} />
        <Stat label="Уникальных документов" value={summary.processed_unique_documents} />
      </section>

      <section className="mt-9">
        <h2 className="mb-1 text-xl font-semibold">
          {view === 'top15' ? 'Финальный TOP-15' : view === 'watchlist' ? 'Темы для наблюдения' : 'Исключённые темы'}
        </h2>
        <p className="mb-4 text-sm text-muted-foreground">
          {view === 'top15'
            ? 'Пятнадцать наиболее перспективных тем по модели и проверенным источникам. Статус карточки показывает, достаточно ли подтверждений для основной выдачи.'
            : view === 'watchlist'
              ? 'Перспективные темы, которым пока не хватает полного покрытия или независимых подтверждений.'
              : 'Зрелые, рекламные, повторные и другие темы, не включённые в финальную выдачу. Для каждой сохранена причина.'}
        </p>
        <div className="grid gap-3">
          {candidates.map((candidate) => (
            <CandidateCard key={candidate.candidate_id} candidate={candidate} reportBase={reportBase} />
          ))}
        </div>
      </section>
    </main>
  )
}

export function CandidateReportPage() {
  const { analysisId, candidateId } = useParams()
  const result = useResource(
    () => analysisId ? api.analysisResult(analysisId) : api.currentResult(),
    `candidate-report:${analysisId ?? 'current'}:${candidateId}`,
  )
  if (result.loading) return <Message>Загружаю отчёт по теме…</Message>
  if (result.error || !result.data) return <Message error={result.error ?? 'Отчёт отсутствует'} />
  const candidate = result.data.candidates.find((item) => item.candidate_id === candidateId)
  if (!candidate) return <Message error="Тема не найдена в сохранённом результате" />
  const backTo = analysisId ? `/analyses/${analysisId}` : '/result'

  return (
    <article className="mx-auto max-w-4xl px-4 py-8 sm:px-8 sm:py-12">
      <Link to={backTo} className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-3.5" /> К результатам анализа
      </Link>
      <div className="mt-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <Badge variant="outline">{labels[candidate.status]}</Badge>
          <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-balance">{candidateDisplayName(candidate)}</h1>
        </div>
        {candidate.top15_rank && <Badge>#{candidate.top15_rank} в TOP-15</Badge>}
      </div>

      <section className="mt-8">
        <h2 className="text-lg font-medium">Краткое резюме</h2>
        <p className="mt-2 leading-relaxed text-muted-foreground">
          {candidateSummary(candidate)}
        </p>
      </section>

      <section className="mt-7">
        <h2 className="text-lg font-medium">Потенциальное преимущество</h2>
        <p className="mt-2 leading-relaxed text-muted-foreground">
          {candidateAdvantage(candidate)
            ? candidateAdvantage(candidate)
            : 'Отдельное подтверждённое преимущество в доступных источниках не найдено.'}
        </p>
      </section>

      <section className="mt-7">
        <h2 className="text-lg font-medium">Кейс-пример</h2>
        {candidate.case_example
          ? <div className="mt-3"><ClaimList title="Подтверждённый пример" claims={[candidate.case_example]} /></div>
          : <p className="mt-2 text-muted-foreground">Проверяемый кейс-пример в доступных источниках не найден.</p>}
      </section>

      <Alert className="mt-7">
        <Info />
        <AlertTitle>Почему присвоен статус «{labels[candidate.status]}»</AlertTitle>
        <AlertDescription>{reasonText(candidate)}</AlertDescription>
      </Alert>

      <div className="mt-8">
        <CandidateDetails candidate={candidate} />
      </div>
    </article>
  )
}

function CandidateCard({ candidate, reportBase }: { candidate: ResultCandidate; reportBase: string }) {
  const [open, setOpen] = useState(false)
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              {candidate.top15_rank && <Badge>#{candidate.top15_rank}</Badge>}
              <Badge variant="outline">{labels[candidate.status]}</Badge>
            </div>
            <CardTitle className="mt-3 text-lg">{candidateDisplayName(candidate)}</CardTitle>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm leading-relaxed text-muted-foreground">
          {candidateSummary(candidate)}
        </p>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <Badge variant="secondary">
            Независимых источников: {candidate.evidence_review.independent_origins}
          </Badge>
          <Badge variant="secondary">
            Проверяемых фрагментов: {candidate.evidence_review.full_candidate_claims}
          </Badge>
        </div>
        <p className="mt-3 border-l-2 pl-3 text-xs text-muted-foreground">{reasonText(candidate)}</p>
        <Button variant="ghost" size="sm" className="mt-3" onClick={() => setOpen(!open)}>
          {open ? <Minus /> : <Plus />} {open ? 'Скрыть объяснение' : 'Почему тема здесь'}
        </Button>
        <Button variant="outline" size="sm" className="mt-3 ml-2" nativeButton={false} render={<Link to={`${reportBase}/candidates/${candidate.candidate_id}`} />}>
          <FileText /> Открыть отчёт
        </Button>
        {open && <CandidateDetails candidate={candidate} />}
      </CardContent>
    </Card>
  )
}

function CandidateDetails({ candidate }: { candidate: ResultCandidate }) {
  return (
    <div className="mt-4 space-y-5 border-t pt-4 text-sm">
      <div>
        <h4 className="font-medium">Оценка модели</h4>
        <p className="mt-2 text-muted-foreground">
          {(candidate.model.score * 100).toFixed(1)} из 100 — сравнительный балл, а не вероятность.
          Рейтинг использует округлённый балл, а внутри одной группы выше ставит темы с большим числом
          независимых источников и проверяемых фрагментов.
        </p>
        <p className="mt-2 text-muted-foreground">
          Сила подтверждения: {candidate.evidence_review.independent_origins} независимых источника и{' '}
          {candidate.evidence_review.full_candidate_claims} проверяемых фрагмента.
        </p>
      </div>
      <FactorList title="Что повысило оценку" factors={candidate.model.top_positive_factors} />
      {meaningfulFactors(candidate.model.top_negative_factors).length > 0 && (
        <FactorList title="Что снизило оценку" factors={meaningfulFactors(candidate.model.top_negative_factors)} />
      )}
      {candidateAdvantage(candidate) && (
        <div>
          <h4 className="font-medium">Потенциальное преимущество</h4>
          <p className="mt-2 text-muted-foreground">{candidateAdvantage(candidate)}</p>
        </div>
      )}
      <ClaimList title="Что подтверждает слабый сигнал" claims={candidate.signal_case} mode="support" />
      <ClaimList title="Что ослабляет вывод" claims={candidate.skeptic_case} mode="skeptic" />
      {candidate.limitations.length > 0 && (
        <div>
          <h4 className="font-medium">Ограничения вывода</h4>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
            {candidate.limitations.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

function FactorList({ title, factors }: { title: string; factors: ResultCandidate['model']['top_positive_factors'] }) {
  return (
    <div>
      <h4 className="font-medium">{title}</h4>
      {factors.length === 0 ? (
        <p className="mt-2 text-muted-foreground">Значимых факторов нет.</p>
      ) : (
        <ul className="mt-2 space-y-1 text-muted-foreground">
          {factors.map((factor) => (
            <li key={`${factor.feature_group}:${factor.feature_name}`}>
              {factor.label_ru}: <span className="tabular-nums">вклад {factor.contribution.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function ClaimList({ title, claims, mode = 'support' }: { title: string; claims: EvidenceClaimView[]; mode?: 'support' | 'skeptic' }) {
  return (
    <div>
      <h4 className="font-medium">{title}</h4>
      {claims.length === 0 ? (
        <p className="mt-2 text-muted-foreground">Проверяемых утверждений нет.</p>
      ) : (
        <ul className="mt-2 space-y-3">
          {claims.map((claim) => (
            <li key={claim.claim_id} className="rounded-lg bg-muted/50 p-3">
              <Badge variant="outline">{claimKindLabel(claim.kind)}</Badge>
              <p className="mt-2">{claimExplanation(claim, mode)}</p>
              <blockquote className="mt-2 border-l-2 pl-3 text-xs text-muted-foreground">
                {claim.quote}
              </blockquote>
              <p className="mt-2 text-xs text-muted-foreground">
                {[sourceTypeLabel(claim.source.source_type), claim.source.published_at, languageLabel(claim.source.language), `доверие: ${trustLabel(claim.source.trust_tier)}`]
                  .filter(Boolean)
                  .join(' · ')}
              </p>
              {(claim.interpretation_generated || claim.source.language !== 'ru') && (
                <p className="mt-1 text-xs text-muted-foreground">Автоматическое резюме на русском языке.</p>
              )}
              {claim.source.url && (
                <a className="mt-2 inline-flex items-center gap-1 text-xs underline" href={claim.source.url} target="_blank" rel="noreferrer">
                  {claim.source.name ?? 'Открыть источник'} <ExternalLink className="size-3" />
                </a>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function meaningfulFactors(factors: ResultCandidate['model']['top_negative_factors']) {
  return factors.filter((factor) => Math.abs(factor.contribution) >= 0.03)
}

function claimExplanation(claim: EvidenceClaimView, mode: 'support' | 'skeptic') {
  if (mode === 'skeptic' && claim.kind === 'promotional_claim') {
    return 'Источник связан с компанией, которая продвигает решение. Такое заявление не считается независимым подтверждением слабого сигнала.'
  }
  if (mode === 'skeptic' && ['adoption', 'standard', 'market'].includes(claim.kind)) {
    return 'Фрагмент указывает на внедрение, сформировавшийся рынок или стандарт. Это признак зрелости технологии, а не раннего слабого сигнала.'
  }
  return plainClaimExplanation(claim.explanation_ru)
}

function reasonText(candidate: ResultCandidate) {
  return reasonDescriptions[candidate.reason] ?? candidate.reason_ru
}

function candidateSummary(candidate: ResultCandidate) {
  if (!candidate.description_ru) return reasonText(candidate)
  const statement = plainClaimExplanation(candidate.description_ru)
  const vague = /не раскрывает|не уточняет|технологическое ядро|наличие |полный контекст|возможные смысловые/iu.test(statement)
  if (!vague) return statement
  const claims = candidate.evidence_review.full_candidate_claims
  const origins = candidate.evidence_review.independent_origins
  return `По теме «${candidateDisplayName(candidate)}» найдено ${claims} проверяемых фрагмента из ${origins} независимых источников. Конкретные факты и ограничения приведены в отчёте.`
}

const russianTechnologyNames: Record<string, string> = {
  gpu: 'Графические процессоры (GPU)',
  gpus: 'Графические процессоры (GPU)',
  npu: 'Нейропроцессоры (NPU)',
  npus: 'Нейропроцессоры (NPU)',
  cpu: 'Центральные процессоры (CPU)',
  cpus: 'Центральные процессоры (CPU)',
  nccl: 'Библиотека коллективных коммуникаций NVIDIA (NCCL)',
}

function candidateDisplayName(candidate: ResultCandidate) {
  return russianTechnologyNames[candidate.canonical_name.trim().toLowerCase()] ?? candidate.canonical_name
}

function candidateAdvantage(candidate: ResultCandidate) {
  const candidates = [candidate.potential_advantage_ru, ...candidate.signal_case.map((claim) => claim.explanation_ru)]
  const benefit = candidates
    .filter((value): value is string => Boolean(value))
    .map(plainClaimExplanation)
    .find((value) => /ускор|повыш|сниж|уменьш|улучш|эффектив|точност|производ|эконом|энергосбереж|безопас|масштаб|преимуществ/iu.test(value))
  return benefit ?? null
}

function compareCandidates(a: ResultCandidate, b: ResultCandidate) {
  const rankA = a.top15_rank ?? Number.POSITIVE_INFINITY
  const rankB = b.top15_rank ?? Number.POSITIVE_INFINITY
  return rankA - rankB || b.model.score - a.model.score || a.canonical_name.localeCompare(b.canonical_name, 'ru')
}

function plainClaimExplanation(value: string) {
  const direct = value
    .replace(/^Цитата (?:прямо )?подтверждает,? что\s+/iu, '')
    .replace(/^Цитата (?:прямо )?(?:подтверждает|показывает|определяет|называет|описывает)\s+/iu, '')
  const about = direct
    .replace(/^Цитата (?:прямо )?сообщает об\s+/iu, 'Есть сведения об ')
    .replace(/^Цитата (?:прямо )?сообщает о\s+/iu, 'Есть сведения о ')
    .replace(/^Цитата\s+/iu, '')
  return about.charAt(0).toLocaleUpperCase('ru-RU') + about.slice(1)
}

function claimKindLabel(kind: string) {
  const labelsByKind: Record<string, string> = {
    novelty: 'Новизна',
    growth: 'Рост',
    research: 'Исследование',
    patent: 'Патент',
    prototype: 'Прототип',
    pilot: 'Пилот',
    investment: 'Инвестиции',
    adoption: 'Внедрение',
    standard: 'Стандарт',
    market: 'Рынок',
    promotional_claim: 'Заявление компании',
  }
  return labelsByKind[kind] ?? 'Доказательство'
}

function sourceTypeLabel(value: string | null) {
  const labelsByType: Record<string, string> = {
    article: 'научная статья',
    journal: 'научный журнал',
    preprint: 'препринт',
    conference: 'материалы конференции',
    patent: 'патент',
    industry_media: 'отраслевое медиа',
    report: 'аналитический отчёт',
    vendor: 'сайт разработчика',
    other: 'публикация',
  }
  return value ? (labelsByType[value] ?? value) : 'тип не указан'
}

function languageLabel(value: string | null) {
  if (value === 'ru') return 'русский'
  if (value === 'en') return 'английский'
  if (value === 'und') return 'язык не определён'
  return value ? `язык: ${value}` : 'язык не указан'
}

function trustLabel(value: string) {
  return value === 'unknown' ? 'не определён' : value
}

function Stat({ label, value }: { label: string; value: number }) {
  return <Card><CardContent><div className="text-2xl font-semibold tabular-nums">{value.toLocaleString('ru-RU')}</div><div className="mt-1 text-xs text-muted-foreground">{label}</div></CardContent></Card>
}

function Message({ children, error }: { children?: React.ReactNode; error?: string }) {
  return <main className="mx-auto max-w-3xl px-4 py-16"><Alert variant={error ? 'destructive' : 'default'}><Info /><AlertTitle>{error ? 'Результат недоступен' : 'NextWave'}</AlertTitle><AlertDescription>{error ?? children}</AlertDescription></Alert><Button variant="outline" className="mt-4" nativeButton={false} render={<Link to="/" />}>Вернуться к запросу</Button></main>
}
