import { useState } from 'react'
import { AlertTriangle, Info, Loader2, RotateCw } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'
import { AnalysisProgress } from '@/components/analysis-progress'
import { TrendCard } from '@/components/trend-card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { api, BUCKETS, type Bucket, type Trend } from '@/lib/api'
import { formatDate, statusLabel } from '@/lib/format'
import { useAnalysis } from '@/lib/hooks'
import { cn } from '@/lib/utils'

const sorts = {
  score: { label: 'по рейтингу', compare: (a: Trend, b: Trend) => b.score - a.score },
  documents: { label: 'по числу документов', compare: (a: Trend, b: Trend) => b.document_count - a.document_count },
  independence: {
    label: 'по независимым источникам',
    compare: (a: Trend, b: Trend) => b.independent_sources - a.independent_sources,
  },
}

export function ResultsPage() {
  const { analysisId } = useParams()
  const { data: analysis, error } = useAnalysis(analysisId)
  const [sort, setSort] = useState<keyof typeof sorts>('score')
  const [bucket, setBucket] = useState<Bucket>('main')

  const trendsIn = (key: Bucket) =>
    (analysis?.trends ?? []).filter((t) => t.bucket === key).sort(sorts[sort].compare)

  if (error) {
    return (
      <Shell>
        <AssistantMessage>
          <Alert variant="destructive">
            <AlertTriangle />
            <AlertTitle>Не удалось получить результат</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
          <Button variant="outline" className="mt-4" render={<Link to="/" />}>
            Новый запрос
          </Button>
        </AssistantMessage>
      </Shell>
    )
  }

  if (!analysis) {
    return (
      <Shell>
        <div className="ml-auto w-2/3 max-w-md">
          <Skeleton className="h-16 w-full rounded-3xl" />
        </div>
        <AssistantMessage>
          <Skeleton className="h-7 w-56" />
          <Skeleton className="mt-4 h-52 w-full rounded-2xl" />
        </AssistantMessage>
      </Shell>
    )
  }

  const running = analysis.status === 'pending' || analysis.status === 'running'

  return (
    <Shell>
      <div className="ml-auto max-w-[85%] rounded-[22px] rounded-br-md bg-muted px-4 py-3 text-[15px] leading-6 sm:max-w-[72%]">
        {analysis.query}
      </div>

      <AssistantMessage>
        <div className="mb-6 flex flex-wrap items-start justify-between gap-3 border-b border-border pb-5">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Исследование
            </p>
            <h1 className="mt-1.5 text-2xl font-semibold tracking-[-0.035em] text-balance">
              «{analysis.query}»
            </h1>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              Запуск {formatDate(analysis.created_at)} · корпус {analysis.corpus_version} · метод{' '}
              {analysis.method_version}
              {analysis.finished_at && ' · сохранённая выдача'}
            </p>
          </div>
          <Badge variant={analysis.status === 'done' ? 'default' : 'secondary'}>
            {statusLabel[analysis.status]}
          </Badge>
        </div>

        {analysis.corpus_version.startsWith('synthetic-') && (
          <Alert className="mb-5">
            <Info />
            <AlertTitle>Демонстрационный режим</AlertTitle>
            <AlertDescription>
              Карточки проверяют интерфейс и API. Они не являются результатами обученной модели.
            </AlertDescription>
          </Alert>
        )}

        {running && <AnalysisProgress analysis={analysis} />}

        {analysis.status === 'error' && (
          <>
            <Alert variant="destructive">
              <AlertTriangle />
              <AlertTitle>Анализ завершился ошибкой</AlertTitle>
              <AlertDescription>{analysis.notice}</AlertDescription>
            </Alert>
            <Retry query={analysis.query} />
          </>
        )}

        {analysis.status === 'empty' && (
          <>
            <Alert>
              <Info />
              <AlertTitle>Кандидатов не найдено</AlertTitle>
              <AlertDescription>{analysis.notice}</AlertDescription>
            </Alert>
            <Button variant="outline" className="mt-4" render={<Link to="/" />}>
              <RotateCw /> Изменить запрос
            </Button>
          </>
        )}

        {analysis.status === 'done' && (
          <Tabs value={bucket} onValueChange={(value) => setBucket(value as Bucket)}>
            <TabsList className="grid h-auto w-full grid-cols-3 rounded-xl p-1">
              {BUCKETS.map((item) => (
                <TabsTrigger key={item.key} value={item.key} className="min-h-9 gap-1.5 px-2 text-xs sm:text-sm">
                  <span className={cn('size-1.5 rounded-full', item.dot)} />
                  <span className="truncate">{item.short}</span>
                  <span className="tabular-nums text-muted-foreground">
                    {analysis.trends.filter((t) => t.bucket === item.key).length}
                  </span>
                </TabsTrigger>
              ))}
            </TabsList>

            {BUCKETS.map((item) => {
              const trends = trendsIn(item.key)
              return (
              <TabsContent key={item.key} value={item.key} className="pt-5">
                <p className="mb-5 text-sm leading-relaxed text-pretty text-muted-foreground">
                  {item.description}
                </p>

                {item.key === 'main' && analysis.notice && (
                  <Alert className="mb-4">
                    <Info />
                    <AlertDescription>{analysis.notice}</AlertDescription>
                  </Alert>
                )}

                {trends.length > 1 && (
                  <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <p className="text-sm text-muted-foreground">
                      {trends.length} кандидатов, отсортированы {sorts[sort].label}
                    </p>
                    <Select value={sort} onValueChange={(v) => setSort(v as keyof typeof sorts)}>
                      <SelectTrigger className="w-full sm:w-56" aria-label="Сортировка">
                        <SelectValue>Сортировка {sorts[sort].label}</SelectValue>
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(sorts).map(([key, { label }]) => (
                          <SelectItem key={key} value={key}>
                            Сортировка {label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="flex flex-col gap-3">
                  {trends.map((trend) => (
                    <TrendCard
                      key={trend.id}
                      trend={trend}
                      to={`/analyses/${analysis.id}/trends/${trend.id}`}
                    />
                  ))}
                </div>

                {trends.length === 0 && (
                  <p className="rounded-lg border border-dashed px-4 py-8 text-center text-sm text-muted-foreground">
                    В этой корзине нет кандидатов по текущему запросу.
                  </p>
                )}
              </TabsContent>
              )
            })}

            <p className="mt-6 text-xs text-muted-foreground">
              Каждая карточка подписана причиной попадания в корзину. Рейтинг задаёт порядок изучения
              кандидатов и не является вероятностью успеха технологии.
            </p>
          </Tabs>
        )}
      </AssistantMessage>
    </Shell>
  )
}

/** Повторный запуск того же запроса: US-06 требует не только объяснения ошибки, но и выхода из неё. */
function Retry({ query }: { query: string }) {
  const [starting, setStarting] = useState(false)
  const navigate = useNavigate()

  async function restart() {
    setStarting(true)
    try {
      const analysis = await api.startAnalysis(query)
      navigate(`/analyses/${analysis.id}`)
    } catch (e) {
      toast.error('Не удалось перезапустить анализ', { description: (e as Error).message })
      setStarting(false)
    }
  }

  return (
    <Button variant="outline" size="sm" className="mt-4" onClick={restart} disabled={starting}>
      {starting ? <Loader2 className="animate-spin" /> : <RotateCw />} Повторить анализ
    </Button>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto flex max-w-4xl flex-col gap-9 px-4 py-8 sm:px-8 sm:py-12">{children}</div>
}

function AssistantMessage({ children }: { children: React.ReactNode }) {
  return (
    <section className="grid grid-cols-[32px_minmax(0,1fr)] gap-3 sm:grid-cols-[36px_minmax(0,1fr)] sm:gap-4">
      <img src="/nextwave_logo.svg" alt="NextWave" className="size-8 object-contain sm:size-9" />
      <div className="min-w-0 pt-1">{children}</div>
    </section>
  )
}
