import { useState } from 'react'
import { ArrowUp, Database, Loader2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { api } from '@/lib/api'
import { useResource } from '@/lib/hooks'

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [starting, setStarting] = useState(false)
  const navigate = useNavigate()
  const coverage = useResource(api.coverage, 'coverage')

  async function start() {
    if (!query.trim() || starting) return
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
    <div className="mx-auto flex min-h-full w-full max-w-4xl flex-col px-4 sm:px-8">
      <div className="flex flex-1 flex-col items-center justify-center py-14 sm:py-20">
        <img src="/nextwave_logo.svg" alt="NextWave" className="mb-5 size-16 object-contain" />
        <h1 className="text-center text-3xl font-semibold tracking-[-0.04em] text-balance sm:text-4xl">
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
          <div className="rounded-[26px] border border-foreground/20 bg-card p-2.5 shadow-[0_12px_45px_rgba(0,0,0,0.08)] transition-shadow focus-within:border-foreground/45 focus-within:shadow-[0_16px_55px_rgba(0,0,0,0.11)] dark:shadow-[0_16px_55px_rgba(0,0,0,0.45)]">
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
              placeholder="Например: агенты и LLM"
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
                className="flex size-9 shrink-0 items-center justify-center rounded-full bg-foreground text-background transition-transform hover:scale-[1.04] disabled:cursor-not-allowed disabled:opacity-25"
              >
                {starting ? <Loader2 className="size-4 animate-spin" /> : <ArrowUp className="size-4" />}
              </button>
            </div>
          </div>
        </form>

        {coverage.data?.examples.length ? (
          <div className="mt-4 flex max-w-2xl flex-wrap justify-center gap-2">
            {coverage.data.examples.slice(0, 4).map((example) => (
              <button
                key={example}
                type="button"
                className="rounded-full border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-foreground/35 hover:bg-muted hover:text-foreground"
                onClick={() => setQuery(example)}
              >
                {example}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <div className="mx-auto mb-6 flex max-w-2xl items-center justify-center gap-2 text-center text-xs text-muted-foreground">
        <Database className="size-3.5 shrink-0" />
        {coverage.loading && <span>Подключаем корпус данных…</span>}
        {coverage.error && <span>Сведения о корпусе временно недоступны</span>}
        {coverage.data && (
          <span>
            {coverage.data.notice ?? 'Версии и источники доступны в результатах анализа.'}
          </span>
        )}
      </div>
    </div>
  )
}
