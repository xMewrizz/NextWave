import type { AnalysisStatus } from '@/lib/api'

const dateFormat = new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' })
const monthFormat = new Intl.DateTimeFormat('ru-RU', { month: 'short', year: '2-digit' })

export const formatDate = (iso: string) => dateFormat.format(new Date(iso))

/** "2026-01" → "янв 26" */
export const formatPeriod = (period: string) => monthFormat.format(new Date(`${period}-01`))

export const percent = (value: number) => Math.round(value * 100)

export const statusLabel: Record<AnalysisStatus, string> = {
  pending: 'В очереди',
  running: 'Выполняется',
  done: 'Готово',
  empty: 'Пустая выдача',
  error: 'Ошибка',
}
