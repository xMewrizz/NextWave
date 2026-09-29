import { useState } from 'react'
import { ArrowUp, Loader2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { AsciiGlobe, AsciiLogo } from '@/components/ascii-art'
import { ModeNotice } from '@/components/mode-notice'
import { api } from '@/lib/api'
import { useResource } from '@/lib/hooks'

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [starting, setStarting] = useState(false)
  const navigate = useNavigate()
  const health = useResource(api.health, 'health')
  const mode = health.data?.mode
  // Сохранённый результат существует только в режиме снимка; в live он бы вводил в заблуждение.
  const result = useResource(
    () => (mode === 'cached_snapshot' ? api.currentResult() : Promise.resolve(null)),
    `current-result:${mode}`,
  )

  async function start() {
    if (starting) return
    if (!query.trim()) {
      toast.error('Введите технологическое направление')
      return
    }
    setStarting(true)
    try {
      const analysis = await api.startAnalysis(query.trim())
      navigate(`/analyses/${analysis.id}`)
    } catch (e) {
      toast.error('Не удалось запустить анализ', { description: (e as Error).message })
      setStarting(false)
    }
  }

  return (
    <div className="relative flex min-h-full flex-col overflow-hidden">
      <AsciiGlobe className="pointer-events-none absolute inset-x-0 bottom-0 h-[44vh] w-full" />
      <div className="relative mx-auto flex w-full max-w-4xl flex-1 flex-col px-4 sm:px-8">
        <div className="flex flex-1 flex-col items-center justify-center pt-14 pb-[30vh] sm:pt-20">
        <AsciiLogo className="mb-6 size-14" />
        <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">NextWave · радар технологий</p>
        <h1 className="mt-3 text-center text-2xl font-normal text-balance sm:text-3xl">
          Что будем исследовать?
        </h1>
        <p className="mt-3 max-w-lg text-center text-sm leading-relaxed text-muted-foreground sm:text-base">
          Опишите технологическое направление — агент изучит корпус, найдёт слабые сигналы и объяснит
          каждый результат.
        </p>

        <form
          className="mt-8 w-full max-w-2xl"
          onSubmit={(event) => {
            event.preventDefault()
            start()
          }}
        >
          <div className="rounded-lg border border-border bg-background/85 p-2.5 backdrop-blur-sm transition-colors focus-within:border-foreground/70">
            <textarea
              value={query}
              rows={2}
              maxLength={200}
              autoFocus
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  start()
                }
              }}
              placeholder={result.data?.summary.source_query ?? 'Технологическое направление'}
              aria-label="Технологическое направление"
              className="block max-h-40 min-h-16 w-full resize-none bg-transparent px-2.5 py-2 text-[15px] leading-6 outline-none placeholder:text-muted-foreground/75"
            />
            <div className="flex items-center justify-between gap-3 px-1">
              <span className="truncate pl-1 text-xs text-muted-foreground">
                Enter — отправить · Shift + Enter — новая строка
              </span>
              <button
                type="submit"
                disabled={!query.trim() || starting}
                aria-label="Начать исследование"
                className="flex size-9 shrink-0 items-center justify-center rounded-md bg-foreground text-background transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-25"
              >
                {starting ? <Loader2 className="size-4 animate-spin" /> : <ArrowUp className="size-4" />}
              </button>
            </div>
          </div>
        </form>

        {health.error && (
          <p role="alert" className="mt-4 max-w-2xl text-center text-xs text-destructive">
            Сервис недоступен: {health.error}
          </p>
        )}
        {mode && <ModeNotice mode={mode} className="mt-4 w-full max-w-2xl" />}
        {result.error && (
          <p role="alert" className="mt-3 max-w-2xl text-center text-xs text-destructive">
            Сохранённый результат недоступен: {result.error}
          </p>
        )}

        {result.data && (
          <button
            type="button"
            className="mt-4 rounded-md border border-border bg-background/85 px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-foreground/60 hover:text-foreground"
            onClick={() => setQuery(result.data!.summary.source_query)}
          >
            Подставить запрос сохранённого результата
          </button>
        )}
        </div>
        {result.data && (
          <p className="relative mx-auto mb-6 max-w-2xl text-center text-xs text-muted-foreground">
            Проверено {result.data.summary.processed_unique_documents.toLocaleString('ru-RU')} документов ·{' '}
            {result.data.summary.candidate_count} кандидатов
          </p>
        )}
      </div>
    </div>
  )
}
