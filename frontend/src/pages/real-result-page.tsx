import { useMemo, useState } from 'react'
import { ExternalLink, Info, Minus, Plus } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { api, type Bucket, type EvidenceClaimView, type ResultBundle, type ResultCandidate } from '@/lib/api'
import { percent } from '@/lib/format'
import { useResource } from '@/lib/hooks'

const labels: Record<Bucket, string> = {
  main: 'Прошли policy',
  watchlist: 'Наблюдение',
  excluded: 'Исключены',
}

export function RealResultPage() {
  const { data, error, loading } = useResource(api.currentResult, 'current-result-page')
  if (loading) return <Message>Загружаю проверенный результат…</Message>
  if (error || !data) return <Message error={error ?? 'Результат отсутствует'} />
  return <ResultView data={data} />
}

export function ResultView({ data }: { data: ResultBundle }) {
  const [bucket, setBucket] = useState<Bucket>('main')
  const candidates = useMemo(
    () =>
      data.candidates
        .filter((candidate) => candidate.status === bucket)
        .sort((a, b) => b.model.score - a.model.score || a.candidate_id.localeCompare(b.candidate_id)),
    [bucket, data],
  )
  const summary = data.summary

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-8 sm:py-12">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
            Реальный зафиксированный анализ
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-balance">
            {summary.source_query}
          </h1>
        </div>
        <Badge variant="secondary">{summary.release_status}</Badge>
      </div>

      <Alert className="mt-6">
        <Info />
        <AlertTitle>Оценка модели не является вероятностью</AlertTitle>
        <AlertDescription>
          Калибровка не выполнена, поэтому интерфейс показывает model_score и порог. Счётчик
          выше 0,75 также относится к score, а не к «уверенности 75%».
        </AlertDescription>
      </Alert>

      <section className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Stat label="Проверено предложений" value={summary.candidate_gate?.evaluated_proposals ?? 0} />
        <Stat label="Кандидатов" value={summary.candidate_count} />
        <Stat label="В TOP-15" value={summary.top15_count} />
        <Stat label="Уникальных документов" value={summary.processed_unique_documents} />
        <Stat label="model_score > 0,75" value={summary.candidates_model_score_gt_075} />
      </section>

      <section className="mt-9">
        <h2 className="text-xl font-semibold">Финальный TOP-15</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Только кандидаты main после Evidence Duel и фиксированной decision policy.
        </p>
        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          {data.top15.map((candidate) => (
            <CandidateCard key={candidate.candidate_id} candidate={candidate} />
          ))}
        </div>
      </section>

      <section className="mt-10">
        <h2 className="mb-4 text-xl font-semibold">Аудит всех кандидатов</h2>
        <Tabs value={bucket} onValueChange={(value) => setBucket(value as Bucket)}>
          <TabsList className="grid h-auto w-full grid-cols-3">
            {(Object.keys(labels) as Bucket[]).map((key) => (
              <TabsTrigger key={key} value={key}>
                {labels[key]} · {summary.status_counts[key] ?? 0}
              </TabsTrigger>
            ))}
          </TabsList>
          {(Object.keys(labels) as Bucket[]).map((key) => (
            <TabsContent key={key} value={key} className="mt-4 grid gap-3 lg:grid-cols-2">
              {candidates.map((candidate) => (
                <CandidateCard key={candidate.candidate_id} candidate={candidate} />
              ))}
            </TabsContent>
          ))}
        </Tabs>
      </section>
    </main>
  )
}

function CandidateCard({ candidate }: { candidate: ResultCandidate }) {
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
            <CardTitle className="mt-3 text-lg">{candidate.canonical_name}</CardTitle>
          </div>
          <span className="text-lg font-semibold tabular-nums">{percent(candidate.model.score)}</span>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm leading-relaxed text-muted-foreground">
          {candidate.description_ru ?? candidate.reason_ru}
        </p>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <Badge variant="secondary">порог {percent(candidate.model.threshold)}</Badge>
          <Badge variant="secondary">origin: {candidate.evidence_review.independent_origins}</Badge>
          <Badge variant="secondary">claims: {candidate.evidence_review.full_candidate_claims}</Badge>
        </div>
        <p className="mt-3 border-l-2 pl-3 text-xs text-muted-foreground">{candidate.reason_ru}</p>
        <Button variant="ghost" size="sm" className="mt-3" onClick={() => setOpen(!open)}>
          {open ? <Minus /> : <Plus />} {open ? 'Скрыть разбор' : 'Показать Evidence Duel'}
        </Button>
        {open && <CandidateDetails candidate={candidate} />}
      </CardContent>
    </Card>
  )
}

function CandidateDetails({ candidate }: { candidate: ResultCandidate }) {
  return (
    <div className="mt-4 space-y-5 border-t pt-4 text-sm">
      <FactorList title="Факторы за слабый сигнал" factors={candidate.model.top_positive_factors} />
      <FactorList title="Факторы против" factors={candidate.model.top_negative_factors} />
      <ClaimList title="Signal case" claims={candidate.signal_case} />
      <ClaimList title="Skeptic case" claims={candidate.skeptic_case} />
      {candidate.limitations.length > 0 && (
        <div>
          <h4 className="font-medium">Ограничения</h4>
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
      <ul className="mt-2 space-y-1 text-muted-foreground">
        {factors.map((factor) => (
          <li key={`${factor.feature_group}:${factor.feature_name}`}>
            {factor.label_ru}: <span className="tabular-nums">{factor.contribution.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function ClaimList({ title, claims }: { title: string; claims: EvidenceClaimView[] }) {
  return (
    <div>
      <h4 className="font-medium">{title}</h4>
      {claims.length === 0 ? (
        <p className="mt-2 text-muted-foreground">Проверяемых утверждений нет.</p>
      ) : (
        <ul className="mt-2 space-y-3">
          {claims.map((claim) => (
            <li key={claim.claim_id} className="rounded-lg bg-muted/50 p-3">
              <p>{claim.explanation_ru}</p>
              <blockquote className="mt-2 border-l-2 pl-3 text-xs text-muted-foreground">
                {claim.quote}
              </blockquote>
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

function Stat({ label, value }: { label: string; value: number }) {
  return <Card><CardContent><div className="text-2xl font-semibold tabular-nums">{value.toLocaleString('ru-RU')}</div><div className="mt-1 text-xs text-muted-foreground">{label}</div></CardContent></Card>
}

function Message({ children, error }: { children?: React.ReactNode; error?: string }) {
  return <main className="mx-auto max-w-3xl px-4 py-16"><Alert variant={error ? 'destructive' : 'default'}><Info /><AlertTitle>{error ? 'Результат недоступен' : 'NextWave'}</AlertTitle><AlertDescription>{error ?? children}</AlertDescription></Alert><Button variant="outline" className="mt-4" nativeButton={false} render={<Link to="/" />}>Вернуться к запросу</Button></main>
}
