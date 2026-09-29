import { AsciiGlobe } from '@/components/ascii-art'
import type { AnalysisJob } from '@/lib/api'
import { cn } from '@/lib/utils'

export function AnalysisProgress({ analysis }: { analysis: AnalysisJob }) {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between gap-4 text-xs text-muted-foreground">
        <span className="truncate">{analysis.stage_label ?? 'Запуск анализа'}…</span>
        <span className="tabular-nums">{Math.round(analysis.progress * 100)}%</span>
      </div>

      <ol className="flex overflow-x-auto pb-2 [scrollbar-width:thin]">
        {analysis.stage_history.map((stage, i) => {
          const done = stage.status === 'complete' || stage.status === 'reused'
          const active = stage.status === 'running'
          return (
            <li key={stage.key} className="min-w-32 flex-1">
              <div className="flex items-center">
                <span
                  className={cn(
                    'size-2.5 shrink-0 rounded-full border border-solid border-muted-foreground',
                    done && 'border-foreground bg-foreground',
                    active && 'animate-pulse border-foreground bg-foreground ring-4 ring-foreground/15',
                  )}
                />
                {i < analysis.stage_history.length - 1 && (
                  <span className={cn('mx-2 flex-1 border-t border-dotted', done && 'border-foreground/70')} />
                )}
              </div>
              <p
                className={cn(
                  'mt-2.5 pr-3 text-xs leading-snug',
                  done && 'text-muted-foreground',
                  active && 'text-foreground',
                  !done && !active && 'text-muted-foreground/60',
                )}
              >
                {stage.label}{stage.status === 'reused' ? ' · сохранено' : ''}
              </p>
            </li>
          )
        })}
      </ol>

      <AsciiGlobe className="-mx-4 h-72 w-[calc(100%+2rem)] sm:h-96" />
    </div>
  )
}
