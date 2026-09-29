const dateFormat = new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' })

export const formatDate = (iso: string) => dateFormat.format(new Date(iso))

/** Как на backend (_normalized_query): пробелы и регистр не отличают запросы. */
export const normalizeQuery = (value: string) => value.split(/\s+/u).filter(Boolean).join(' ').toLowerCase()
