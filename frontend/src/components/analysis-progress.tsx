import { Check, Loader2 } from 'lucide-react'
import { api, type Analysis } from '@/lib/api'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { useResource } from '@/lib/hooks'
import { cn } from '@/lib/utils'

export function AnalysisProgress({ analysis }: { analysis: Analysis }) {
  const stages = useResource(api.stages, 'stages')
  const current = stages.data?.findIndex((s) => s.label === analysis.stage) ?? -1

  return (
    <div className="flex flex-col gap-5">
      <div className="rounded-2xl border border-foreground/15 bg-card p-4 sm:p-5">
        <div className="mb-3 flex items-center justify-between gap-4 text-sm">
          <span className="flex min-w-0 items-center gap-2 font-medium">
            <Loader2 className="size-3.5 shrink-0 animate-spin" />
            <span className="truncate">
            {analysis.stage ?? 'Запуск анализа'}
            </span>
          </span>
          <span className="tabular-nums text-muted-foreground">{Math.round(analysis.progress * 100)}%</span>
        </div>
        <Progress value={analysis.progress * 100} className="[&_[data-slot=progress-track]]:h-1.5" />
      </div>

      <ol className="grid gap-1 rounded-2xl border border-border p-2 sm:p-3">
        {stages.data?.map((stage, i) => {
          const done = current > i
          const active = current === i
          return (
            <li
              key={stage.key}
              className={cn(
                'flex min-h-9 items-center gap-3 rounded-xl px-2.5 text-sm transition-colors',
                done && 'text-muted-foreground',
                active && 'bg-muted font-medium text-foreground',
                !done && !active && 'text-muted-foreground/50',
              )}
            >
              <span
                className={cn(
                  'flex size-5 shrink-0 items-center justify-center rounded-full border text-[10px]',
                  done && 'border-foreground/30 bg-foreground text-background',
                  active && 'border-foreground bg-foreground text-background',
                )}
              >
                {done ? <Check className="size-3" /> : i + 1}
              </span>
              {stage.label}
            </li>
          )
        })}
      </ol>

      <div className="flex flex-col gap-3" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-20 w-full rounded-2xl" />
        ))}
      </div>
    </div>
  )
}
