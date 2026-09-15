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
import { BUCKETS, FACTOR_LABELS, SOURCE_LABELS, trendScore, type ScoreFactor } from '@/lib/api'
import { formatDate, formatPeriod, percent } from '@/lib/format'
import { useAnalysis } from '@/lib/hooks'
import { cn } from '@/lib/utils'

export function TrendPage() {
  const { analysisId, trendId } = useParams()
  const { data: analysis, error } = useAnalysis(analysisId)
  const trend = analysis?.trends.find((item) => item.candidate_id === trendId)
  const backTo = `/analyses/${analysisId}`

  if (error || (analysis && !trend)) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8 sm:px-8 sm:py-12">
        <Alert variant="destructive">
          <AlertTriangle />
          <AlertTitle>Тренд не найден</AlertTitle>
          <AlertDescription>{error ?? 'Карточки нет в этой выдаче.'}</AlertDescription>
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
          <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">{trend.summary}</p>

          <Alert className="mt-5">
            <ListFilter className={bucket.accent} />
            <AlertTitle>Почему кандидат попал в «{bucket.label}»</AlertTitle>
            <AlertDescription>{trend.explanation}</AlertDescription>
          </Alert>

          <Separator className="my-8" />

          <h2 className="mb-4 text-lg font-medium">Факторы оценки</h2>
          <div className="flex flex-col gap-3">
            {trend.factors.map((factor) => (
              <FactorBar key={factor.key} factor={factor} />
            ))}
          </div>

          <Separator className="my-8" />

          <div className="grid gap-4 sm:grid-cols-2">
            <InfoCard icon={Target} title="Проблема" text={trend.problem} />
            <InfoCard icon={Lightbulb} title="Преимущество" text={trend.advantage} />
          </div>

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

          {trend.hypothesis && (
            <Alert className="mt-4">
              <Lightbulb />
              <AlertTitle>Гипотеза применения, не подтверждённый факт</AlertTitle>
              <AlertDescription>{trend.hypothesis}</AlertDescription>
            </Alert>
          )}

          <Separator className="my-8" />

          <h2 className="mb-1 text-lg font-medium">Динамика в корпусе</h2>
          <p className="mb-4 text-sm text-muted-foreground">
            Первое найденное упоминание в корпусе — {formatPeriod(trend.first_seen)}. Это не дата появления
            технологии, а граница доступных данных.
          </p>
          <TrendChart timeline={trend.timeline} />

          <Separator className="my-8" />

          <h2 className="mb-4 text-lg font-medium">Источники</h2>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Материал</TableHead>
                <TableHead className="w-32">Тип</TableHead>
                <TableHead className="w-28">Дата</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trend.evidence.map((source) => (
                <TableRow key={source.evidence_id}>
                  <TableCell>
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 hover:underline"
                    >
                      {source.title} <ExternalLink className="size-3 shrink-0 text-muted-foreground" />
                    </a>
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

          <Separator className="my-8" />

          <h2 className="mb-3 text-lg font-medium">Ограничения</h2>
          <ul className="flex list-disc flex-col gap-1.5 pl-5 text-sm text-muted-foreground">
            {trend.limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
        </div>

        <aside className="lg:sticky lg:top-20 lg:self-start">
          <Card>
            <CardContent className="flex flex-col gap-3 text-sm">
              <Fact label="Рейтинг" value={`${percent(trendScore(trend))} из 100`} />
              <Fact label="Документов в теме" value={String(trend.document_count)} />
              <Fact
                label="Независимых источников"
                value={String(trend.features.independent_source_count)}
              />
              <Fact label="Первое упоминание" value={formatPeriod(trend.first_seen)} />
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
