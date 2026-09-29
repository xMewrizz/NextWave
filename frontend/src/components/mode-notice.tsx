import { Archive, Radio } from 'lucide-react'
import type { JobMode } from '@/lib/api'
import { cn } from '@/lib/utils'

const modes = {
  cached_snapshot: {
    icon: Archive,
    title: 'Проверенный снимок',
    text: 'Показан ранее сохранённый результат реального анализа, целостность которого проверена по SHA-256. Внешние источники не запрашиваются; доступен только зафиксированный запрос.',
  },
  live: {
    icon: Radio,
    title: 'Живой анализ',
    text: 'Запрос обрабатывается по открытым источникам прямо сейчас: это занимает время и расходует квоты внешних сервисов.',
  },
} as const

export function ModeNotice({ mode, className }: { mode: JobMode; className?: string }) {
  const { icon: Icon, title, text } = modes[mode]
  return (
    <p
      className={cn(
        'flex items-start gap-2 rounded-md border border-border bg-background/85 px-3 py-2 text-xs leading-relaxed text-muted-foreground',
        className,
      )}
    >
      <Icon className="mt-0.5 size-3.5 shrink-0" />
      <span>
        <span className="font-medium text-foreground">Режим: {title}.</span> {text}
      </span>
    </p>
  )
}
