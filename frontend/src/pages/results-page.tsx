import { AlertTriangle, Loader2, RotateCw } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'
import { AnalysisProgress } from '@/components/analysis-progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { api } from '@/lib/api'
import { formatDate } from '@/lib/format'
import { useAnalysisJob, useResource } from '@/lib/hooks'
import { ResultView } from '@/pages/real-result-page'

export function ResultsPage() {
  const { analysisId } = useParams()
  const { data: job, error } = useAnalysisJob(analysisId)

  if (error) {
    return <JobError title="Не удалось получить исследование" message={error} />
  }
  if (!job) {
    return (
      <Shell>
        <Skeleton className="h-24 w-full rounded-2xl" />
        <Skeleton className="h-72 w-full rounded-2xl" />
      </Shell>
    )
  }
  if (job.status === 'complete') return <CompletedJob jobId={job.id} />

  return (
    <Shell>
      <div className="ml-auto max-w-[85%] rounded-[22px] rounded-br-md bg-muted px-4 py-3 text-[15px] leading-6 sm:max-w-[72%]">
        {job.query}
      </div>
      <AssistantMessage>
        <header className="mb-6 flex flex-wrap items-start justify-between gap-3 border-b border-border pb-5">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Исследование
            </p>
            <h1 className="mt-1.5 text-2xl font-semibold tracking-[-0.035em] text-balance">
              «{job.query}»
            </h1>
            <p className="mt-2 text-xs text-muted-foreground">
              Запуск {formatDate(job.created_at)} · сохраняемый analysis job
            </p>
          </div>
          <Badge variant="secondary">{job.status === 'error' ? 'Ошибка' : 'Выполняется'}</Badge>
        </header>

        {job.status === 'error' ? (
          <>
            <Alert variant="destructive">
              <AlertTriangle />
              <AlertTitle>Анализ завершился ошибкой</AlertTitle>
              <AlertDescription>{job.error ?? 'Причина не сохранена'}</AlertDescription>
            </Alert>
            <Retry query={job.query} />
          </>
        ) : (
          <AnalysisProgress analysis={job} />
        )}
      </AssistantMessage>
    </Shell>
  )
}

function CompletedJob({ jobId }: { jobId: string }) {
  const result = useResource(() => api.analysisResult(jobId), `analysis-result:${jobId}`)
  if (result.loading) {
    return (
      <Shell>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Загружаю сохранённый результат…
        </div>
      </Shell>
    )
  }
  if (result.error || !result.data) {
    return (
      <JobError
        title="Результат job повреждён или недоступен"
        message={result.error ?? 'Backend не вернул result.json'}
      />
    )
  }
  return <ResultView data={result.data} />
}

function Retry({ query }: { query: string }) {
  const navigate = useNavigate()
  async function restart() {
    try {
      const job = await api.startAnalysis(query)
      navigate(`/analyses/${job.id}`)
    } catch (error) {
      toast.error('Не удалось перезапустить анализ', {
        description: (error as Error).message,
      })
    }
  }
  return (
    <Button variant="outline" size="sm" className="mt-4" onClick={restart}>
      <RotateCw /> Повторить анализ
    </Button>
  )
}

function JobError({ title, message }: { title: string; message: string }) {
  return (
    <Shell>
      <Alert variant="destructive">
        <AlertTriangle />
        <AlertTitle>{title}</AlertTitle>
        <AlertDescription>{message}</AlertDescription>
      </Alert>
      <Button variant="outline" className="w-fit" render={<Link to="/" />}>
        Новый запрос
      </Button>
    </Shell>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  return <main className="mx-auto flex max-w-4xl flex-col gap-9 px-4 py-8 sm:px-8 sm:py-12">{children}</main>
}

function AssistantMessage({ children }: { children: React.ReactNode }) {
  return (
    <section className="grid grid-cols-[32px_minmax(0,1fr)] gap-3 sm:grid-cols-[36px_minmax(0,1fr)] sm:gap-4">
      <img src="/nextwave_logo.svg" alt="NextWave" className="size-8 object-contain sm:size-9" />
      <div className="min-w-0 pt-1">{children}</div>
    </section>
  )
}
