import {
  AlertTriangle,
  ArrowLeft,
  ExternalLink,
  FlaskConical,
  Lightbulb,
  ListFilter,
  Target,
} from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { TrendChart } from '@/components/trend-chart'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  BUCKETS, FACTOR_LABELS, SOURCE_LABELS, trendScore,
  type CandidateFeatures, type ScoreFactor,
} from '@/lib/api'
import { formatDate, formatPeriod, percent } from '@/lib/format'
import { useAnalysis } from '@/lib/hooks'
import { cn } from '@/lib/utils'

export function TrendPage() {
  const { analysisId, trendId } = useParams()
  const { data: analysis, error } = useAnalysis(analysisId)
  const trend = analysis?.trends.find((item) => item.candidate_id === trendId)
  const backTo = `/analyses/${analysisId}`

  const running = analysis?.status === 'pending' || analysis?.status === 'running'

  if (error || (analysis && !running && !trend)) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8 sm:px-8 sm:py-12">
        <Alert variant="destructive">
          <AlertTriangle />
          <AlertTitle>Тренд не найден</AlertTitle>
          <AlertDescription>{error ?? analysis?.notice ?? 'Карточки нет в этой выдаче.'}</AlertDescription>
        </Alert>
        <Button variant="outline" className="mt-4" render={<Link to={backTo} />}>
          <ArrowLeft /> К списку
        </Button>
      </div>
    )
  }

  if (!trend) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8 sm:px-8 sm:py-12">
        <Skeleton className="h-8 w-72" />
        <Skeleton className="mt-4 h-64 w-full" />
      </div>
    )
  }

  const bucket = BUCKETS.find((item) => item.key === trend.status)!

  return (
    <article className="mx-auto max-w-5xl px-4 py-8 sm:px-8 sm:py-12">
      <Link
        to={backTo}
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3.5" /> К списку кандидатов
      </Link>

      <div className="mt-3 grid gap-10 lg:grid-cols-[1fr_260px]">
        <div className="min-w-0">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="gap-1.5">
              <span className={cn('size-1.5 rounded-full', bucket.dot)} />
              {bucket.label}
            </Badge>
            <Badge variant="outline">№{trend.rank} в корзине</Badge>
          </div>
          <h1 className="text-3xl font-semibold tracking-[-0.04em] text-balance sm:text-4xl">
            {trend.canonical_name}
          </h1>
          {trend.summary && (
            <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">{trend.summary}</p>
          )}

          <Alert className="mt-5">
            <ListFilter className={bucket.accent} />
            <AlertTitle>Почему кандидат попал в «{bucket.label}»</AlertTitle>
            <AlertDescription>
              {trend.explanation}
              {trend.exclusion_reason && <p>Причина исключения: {EXCLUSION_LABELS[trend.exclusion_reason]}</p>}
            </AlertDescription>
          </Alert>

          <Separator className="my-8" />

          <h2 className="mb-4 text-lg font-medium">Признаки модели</h2>
          <dl className="grid gap-3 text-sm">
            {Object.entries(trend.features).map(([key, value]) => (
              <div key={key} className="flex justify-between gap-4">
                <dt className="text-muted-foreground">{FEATURE_LABELS[key as keyof CandidateFeatures]}</dt>
                <dd>{featureValue(value)}</dd>
              </div>
            ))}
          </dl>
          {trend.factors.length > 0 && <h2 className="mt-6 mb-4 text-lg font-medium">Факторы оценки</h2>}
          <div className="flex flex-col gap-3">
            {trend.factors.map((factor) => (
              <FactorBar key={factor.key} factor={factor} />
            ))}
          </div>

          <Separator className="my-8" />

          <div className="grid gap-4 sm:grid-cols-2">
            {trend.problem && <InfoCard icon={Target} title="Проблема" text={trend.problem} />}
            {trend.advantage && <InfoCard icon={Lightbulb} title="Преимущество" text={trend.advantage} />}
          </div>

          {trend.use_case && (
            <Card className="mt-4">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <FlaskConical className="size-4 text-muted-foreground" />
                  Кейс: {trend.use_case.title}
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                <p>{trend.use_case.description}</p>
                {trend.use_case.organization && (
                  <p className="mt-2 text-xs">Источник кейса: {trend.use_case.organization}</p>
                )}
                <a
                  href={trend.use_case.url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-3 inline-flex items-center gap-1 font-medium underline underline-offset-4"
                >
                  Открыть подтверждение <ExternalLink className="size-3.5" />
                </a>
              </CardContent>
            </Card>
          )}

          {trend.hypothesis && (
            <Alert className="mt-4">
              <Lightbulb />
              <AlertTitle>Гипотеза применения, не подтверждённый факт</AlertTitle>
              <AlertDescription>{trend.hypothesis}</AlertDescription>
            </Alert>
          )}

          <Separator className="my-8" />

          {(trend.first_seen || trend.timeline.length > 0) && (
            <>
            <h2 className="mb-1 text-lg font-medium">Динамика в корпусе</h2>
            {trend.first_seen && <p className="mb-4 text-sm text-muted-foreground">
              Первое найденное упоминание в корпусе — {formatPeriod(trend.first_seen)}. Это не дата появления
              технологии, а граница доступных данных.
            </p>}
            {trend.timeline.length > 0 && <TrendChart timeline={trend.timeline} />}
            </>
          )}

          <Separator className="my-8" />

          <h2 className="mb-4 text-lg font-medium">Evidence Duel: источники и аргументы</h2>
          {trend.evidence.length === 0 && <p className="mb-4 text-sm text-muted-foreground">Анализатор не передал доказательств.</p>}
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Материал</TableHead>
                <TableHead className="w-32">Тип</TableHead>
                <TableHead className="w-28">Дата</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trend.evidence.map((source, index) => (
                <TableRow key={`${source.evidence_id}-${index}`}>
                  <TableCell>
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 hover:underline"
                    >
                      {source.title} <ExternalLink className="size-3 shrink-0 text-muted-foreground" />
                    </a>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {source.direction === 'support' ? 'За' : source.direction === 'counter' ? 'Против' : 'Направление не указано'}
                      {' · '}{source.language} · доверенность: {TRUST_LABELS[source.trust_level]}
                    </p>
                    {source.generated_summary && <Badge variant="outline">Генеративное резюме</Badge>}
                    {source.excerpt && <blockquote className="mt-2 border-l-2 pl-3 text-sm">{source.excerpt}</blockquote>}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {SOURCE_LABELS[source.source_type]}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {source.published_at ? formatDate(source.published_at) : 'дата неизвестна'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {trend.limitations.length > 0 && (
            <>
              <Separator className="my-8" />
            <h2 className="mb-3 text-lg font-medium">Ограничения</h2>
            <ul className="flex list-disc flex-col gap-1.5 pl-5 text-sm text-muted-foreground">
              {trend.limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
            </>
          )}
        </div>

        <aside className="lg:sticky lg:top-20 lg:self-start">
          <Card>
            <CardContent className="flex flex-col gap-3 text-sm">
              <Fact label="Оценка модели" value={`${percent(trendScore(trend))} из 100`} />
              <Fact label="Модель" value={trend.prediction.model_version} />
              <Fact label="Признаки" value={trend.prediction.feature_version} />
              {trend.document_count !== null && <Fact label="Документов в теме" value={String(trend.document_count)} />}
              <Fact
                label="Независимых источников"
                value={String(trend.features.independent_source_count)}
              />
              {trend.first_seen && <Fact label="Первое упоминание" value={formatPeriod(trend.first_seen)} />}
              {analysis && (
                <>
                  <Separator />
                  <Fact label="Корпус" value={analysis.corpus_version} />
                  <Fact label="Метод" value={analysis.method_version} />
                </>
              )}
            </CardContent>
          </Card>
          <Button variant="outline" className="mt-3 w-full" render={<Link to="/methodology" />}>
            Как считается рейтинг
          </Button>
        </aside>
      </div>
    </article>
  )
}

function FactorBar({ factor }: { factor: ScoreFactor }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            className="w-full cursor-help rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-muted/60"
          />
        }
      >
        <div className="mb-1.5 flex items-baseline justify-between text-sm">
          <span>{FACTOR_LABELS[factor.key]}</span>
          <span className="tabular-nums text-muted-foreground">{percent(factor.value)}</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{ width: `${percent(factor.value)}%` }}
          />
        </div>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{factor.explanation}</TooltipContent>
    </Tooltip>
  )
}

function InfoCard({ icon: Icon, title, text }: { icon: typeof Target; title: string; text: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Icon className="size-4 text-muted-foreground" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-pretty text-muted-foreground">{text}</CardContent>
    </Card>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium tabular-nums">{value}</span>
    </div>
  )
}

const FEATURE_LABELS: Record<keyof CandidateFeatures, string> = {
  stage: 'Стадия развития',
  independent_source_count: 'Независимые источники',
  source_type_diversity: 'Типы источников',
  independent_actor_count: 'Независимые участники',
  publication_momentum: 'Динамика публикаций',
  patent_momentum: 'Динамика патентов',
  evidence_recency_days: 'Давность доказательств, дней',
  mass_adoption: 'Массовое внедрение',
  formed_market: 'Сформированный рынок',
  industry_standard: 'Отраслевой стандарт',
  promotional_source_share: 'Доля рекламных источников',
}

const STAGE_LABELS: Record<string, string> = {
  research: 'Исследование', prototype: 'Прототип', pilot: 'Пилот',
  early_adoption: 'Раннее внедрение', mass_adoption: 'Массовое внедрение', unknown: 'Неизвестно',
}
const TRUST_LABELS = { high: 'высокая', medium: 'средняя', low: 'низкая', unknown: 'неизвестна' }
const EXCLUSION_LABELS = {
  mature: 'Зрелая технология', mass_adoption: 'Массовое внедрение',
  industry_standard: 'Отраслевой стандарт', marketing_hype: 'Маркетинговый хайп',
  insufficient_trust: 'Недостаточная достоверность', irrelevant: 'Нерелевантность', duplicate: 'Дубликат',
}

function featureValue(value: CandidateFeatures[keyof CandidateFeatures]): string {
  if (value === null) return 'Неизвестно'
  if (typeof value === 'boolean') return value ? 'Да' : 'Нет'
  if (typeof value === 'string') return STAGE_LABELS[value] ?? value
  return String(value)
}
