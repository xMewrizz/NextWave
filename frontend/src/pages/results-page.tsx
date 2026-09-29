import { useState } from 'react'
import { AlertTriangle, Loader2, RotateCw } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { toast } from 'sonner'
import { AnalysisProgress } from '@/components/analysis-progress'
import { AsciiLogo } from '@/components/ascii-art'
import { ModeNotice } from '@/components/mode-notice'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { api, type AnalysisJob } from '@/lib/api'
import { formatDate, normalizeQuery } from '@/lib/format'
import { useAnalysisJob, useResource } from '@/lib/hooks'
import { ResultView } from '@/pages/real-result-page'

export function ResultsPage() {
  const { analysisId } = useParams()
  const { data, error, restart } = useAnalysisJob(analysisId)
  const job = data?.id === analysisId ? data : null

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
  if (job.status === 'complete') return <CompletedJob key={job.id} job={job} />

  return (
    <Shell>
      <div className="ml-auto max-w-[85%] rounded-lg border border-border px-4 py-2.5 text-[15px] leading-6 sm:max-w-[72%]">
        {job.query}
      </div>
      <AssistantMessage>
        <header className="mb-6 flex flex-wrap items-start justify-between gap-3 border-b border-border pb-5">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Исследование
            </p>
            <h1 className="mt-1.5 text-2xl font-normal text-balance">
              «{job.query}»
            </h1>
            <p className="mt-2 text-xs text-muted-foreground">
              Запуск {formatDate(job.created_at)} · задание сохранено и продолжится после перезагрузки страницы
            </p>
          </div>
          <Badge variant="secondary">
            {job.status === 'error' ? 'Ошибка' : job.status === 'pending' ? 'В очереди' : 'Выполняется'}
          </Badge>
        </header>
        <ModeNotice mode={job.mode} className="mb-6" />

        {job.status === 'error' ? (
          <>
            <Alert variant="destructive">
              <AlertTriangle />
              <AlertTitle>Анализ завершился ошибкой</AlertTitle>
              <AlertDescription>
                <p>{errorHint(job)}</p>
                <p className="mt-2 break-words font-mono text-xs opacity-80">{job.error ?? 'Причина не сохранена'}</p>
              </AlertDescription>
            </Alert>
            <Retry jobId={job.id} onRestart={restart} />
          </>
        ) : (
          <AnalysisProgress analysis={job} />
        )}
      </AssistantMessage>
    </Shell>
  )
}

function errorHint(job: AnalysisJob) {
  if (job.mode === 'cached_snapshot') {
    return 'Режим проверенного снимка отвечает только на запрос, для которого сохранён результат. Для другого запроса нужен живой анализ: его включает администратор сервера.'
  }
  if (job.error?.includes('_API_KEY')) {
    return 'На сервере не заданы ключи внешних сервисов, живой анализ не может стартовать. Укажите ключи в .env и перезапустите backend.'
  }
  return 'Живой анализ остановился. Можно повторить: завершённые стадии сохранены и не пересчитываются.'
}

function CompletedJob({ job }: { job: AnalysisJob }) {
  const result = useResource(() => api.analysisResult(job.id), `analysis-result:${job.id}`)
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
        title="Результат недоступен"
        message={result.error ?? 'Backend не вернул результат анализа.'}
      />
    )
  }
  if (normalizeQuery(result.data.query.text) !== normalizeQuery(job.query)) {
    return (
      <JobError
        title="Результат относится к другому запросу"
        message={`Запрошено «${job.query}», а сохранённый результат построен для «${result.data.query.text}». Он не показан.`}
      />
    )
  }
  return <ResultView data={result.data} mode={job.mode} />
}

function Retry({ jobId, onRestart }: { jobId: string; onRestart: () => void }) {
  const [busy, setBusy] = useState(false)
  async function retry() {
    setBusy(true)
    try {
      await api.retryAnalysis(jobId)
      onRestart()
    } catch (error) {
      toast.error('Не удалось повторить анализ', { description: (error as Error).message })
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="mt-4 flex flex-wrap gap-2">
      <Button variant="outline" size="sm" disabled={busy} onClick={retry}>
        {busy ? <Loader2 className="animate-spin" /> : <RotateCw />} Повторить анализ
      </Button>
      <Button variant="ghost" size="sm" nativeButton={false} render={<Link to="/" />}>
        Новый запрос
      </Button>
    </div>
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
      <Button variant="outline" className="w-fit" nativeButton={false} render={<Link to="/" />}>
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
      <AsciiLogo className="size-9" />
      <div className="min-w-0 pt-1">{children}</div>
    </section>
  )
}
