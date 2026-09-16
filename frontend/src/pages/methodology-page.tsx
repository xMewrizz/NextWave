import { BookOpen } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { api, BUCKETS } from '@/lib/api'
import { formatDate } from '@/lib/format'
import { useResource } from '@/lib/hooks'
import { cn } from '@/lib/utils'

const thresholds = [
  { key: 'novelty_min', label: 'Минимальная новизна', scale: 'percent', bucket: 'Отсеяны' },
  { key: 'growth_min', label: 'Минимальный рост', scale: 'percent', bucket: 'Отсеяны' },
  { key: 'evidence_min', label: 'Минимальная доказательная база', scale: 'percent', bucket: 'Наблюдение' },
  { key: 'independent_min', label: 'Независимых источников', scale: 'count', bucket: 'Наблюдение' },
  { key: 'documents_min', label: 'Документов по теме', scale: 'count', bucket: 'Наблюдение' },
]

const factors = [
  { name: 'Рост', check: 'Увеличивается ли доля документов темы в сопоставимых временных окнах' },
  { name: 'Новизна', check: 'Отличается ли тема от исторически представленных направлений' },
  { name: 'Независимость', check: 'Есть ли подтверждения, не сводящиеся к одной публикации или одному событию' },
  { name: 'Доказательная база', check: 'Есть ли содержательные документы и проверяемое исследование' },
]

const rules = [
  'Рейтинг — относительная оценка силы сигнала для порядка изучения, а не вероятность успеха технологии.',
  'Перепечатки одного материала объединяются и не считаются независимыми подтверждениями.',
  'Документы без даты публикации не участвуют в расчёте динамики; дата загрузки её не подменяет.',
  'Тема с недостаточной историей в корпусе не получает высокий балл новизны и не попадает в основной рейтинг.',
]

const limitations = [
  'Пробел в покрытии источников не означает отсутствия исследований по теме.',
  'Первое упоминание — первая найденная дата в корпусе, а не дата появления технологии.',
  'Выдача — набор кандидатов для проверки аналитиком, а не готовый вывод.',
  'Применение в банке, предложенное командой, отделено от подтверждённых свойств технологии и помечено как гипотеза.',
]

export function MethodologyPage() {
  const stages = useResource(api.stages, 'stages')
  const coverage = useResource(api.coverage, 'coverage')

  return (
    <article className="mx-auto max-w-3xl px-4 py-10 sm:px-8 sm:py-14">
      <div className="flex size-10 items-center justify-center rounded-full bg-foreground text-background">
        <BookOpen className="size-4.5" />
      </div>
      <p className="mt-5 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
        О методе
      </p>
      <h1 className="mt-2 text-4xl font-semibold tracking-[-0.045em]">Методология</h1>
      <p className="mt-4 max-w-2xl text-pretty text-base leading-relaxed text-muted-foreground">
        Как сервис превращает открытые публикации в список кандидатов и что означает рейтинг.
      </p>

      <Separator className="my-8" />

      <h2 className="mb-4 text-lg font-medium">Этапы анализа</h2>
      {stages.loading ? (
        <Skeleton className="h-40 w-full" />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {stages.data?.map((stage, i) => (
            <Card key={stage.key}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <span className="flex size-5 items-center justify-center rounded-full bg-foreground text-[10px] text-background">
                    {i + 1}
                  </span>
                  {stage.label}
                </CardTitle>
              </CardHeader>
            </Card>
          ))}
        </div>
      )}

      <Separator className="my-8" />

      <h2 className="mb-4 text-lg font-medium">Основания рейтинга</h2>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-48">Фактор</TableHead>
            <TableHead>Что проверяем</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {factors.map((factor) => (
            <TableRow key={factor.name}>
              <TableCell className="font-medium">{factor.name}</TableCell>
              <TableCell className="text-muted-foreground">{factor.check}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="mt-3 text-xs text-muted-foreground">
        В демонстрационном контуре веса проверяют контракт интерфейса. Измеренные веса и пороги появятся
        после обучения и cross-validation в V1-06.
      </p>

      <Separator className="my-8" />

      <h2 className="mb-1 text-lg font-medium">Три корзины выдачи</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        Кандидат проходит пороги по очереди: сначала проверка на новизну и зарождаемость, затем на
        доказательную базу. Первый непройденный порог определяет корзину, и он же попадает в причину на
        карточке.
      </p>
      <div className="grid gap-3">
        {BUCKETS.map((bucket) => (
          <Card key={bucket.key}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <span className={cn('size-1.5 rounded-full', bucket.dot)} />
                {bucket.label}
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-pretty text-muted-foreground">
              {bucket.description}
            </CardContent>
          </Card>
        ))}
      </div>

      {coverage.data && (
        <>
          <h3 className="mt-6 mb-3 text-sm font-medium">Действующие пороги</h3>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Порог</TableHead>
                <TableHead className="w-24">Значение</TableHead>
                <TableHead className="w-32">Не прошёл —</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {thresholds.map((item) => (
                <TableRow key={item.key}>
                  <TableCell>{item.label}</TableCell>
                  <TableCell className="tabular-nums">
                    {item.scale === 'percent'
                      ? Math.round((coverage.data!.thresholds[item.key] ?? 0) * 100)
                      : coverage.data!.thresholds[item.key]}
                  </TableCell>
                  <TableCell className="text-muted-foreground">{item.bucket}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="mt-3 text-xs text-muted-foreground">
            Пороги меняются вместе с версией метода ({coverage.data.method_version}), поэтому выдача
            остаётся воспроизводимой: тот же корпус и та же версия метода дают то же распределение по
            корзинам.
          </p>
        </>
      )}

      <Separator className="my-8" />

      <h2 className="mb-3 text-lg font-medium">Правила отбора</h2>
      <ul className="flex list-disc flex-col gap-1.5 pl-5 text-sm text-muted-foreground">
        {rules.map((rule) => (
          <li key={rule}>{rule}</li>
        ))}
      </ul>

      <Separator className="my-8" />

      <h2 className="mb-3 text-lg font-medium">Данные и воспроизводимость</h2>
      {coverage.data ? (
        <Card>
          <CardContent className="grid gap-2 text-sm sm:grid-cols-2">
            <Row label="Версия корпуса" value={coverage.data.corpus_version} />
            <Row label="Версия метода" value={coverage.data.method_version} />
            <Row label="Документов" value={String(coverage.data.document_count)} />
            <Row label="Обновлён" value={formatDate(coverage.data.updated_at)} />
            <Row
              label="Период публикаций"
              value={`${formatDate(coverage.data.documents_from)} — ${formatDate(coverage.data.documents_to)}`}
            />
            <Row label="Направления" value={coverage.data.directions.join(', ')} />
          </CardContent>
        </Card>
      ) : (
        <Skeleton className="h-24 w-full" />
      )}
      <p className="mt-3 text-xs text-muted-foreground">
        Для каждого анализа сохраняются запрос, версия корпуса и метода. Повтор на тех же данных с той же
        конфигурацией даёт тот же порядок результатов.
      </p>

      <Separator className="my-8" />

      <h2 className="mb-3 text-lg font-medium">Ограничения</h2>
      <ul className="flex list-disc flex-col gap-1.5 pl-5 text-sm text-muted-foreground">
        {limitations.map((limitation) => (
          <li key={limitation}>{limitation}</li>
        ))}
      </ul>
    </article>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium">{value}</span>
    </div>
  )
}
