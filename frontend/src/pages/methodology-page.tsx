import { BookOpen, CheckCircle2 } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

const stages = [
  {
    title: 'Понимание запроса',
    text: 'Система уточняет предметную область и фиксирует дату, на которую разрешено смотреть данные. Будущие публикации не используются.',
  },
  {
    title: 'Поиск источников',
    text: 'OpenAlex даёт научные публикации. Media Cloud помогает при первичном поиске тем. После первичного отбора Exa ищет отраслевые и веб-источники по каждой теме.',
  },
  {
    title: 'Выделение и объединение тем',
    text: 'Из документов извлекаются названия технологий. Явные варианты написания и дубли объединяются, чтобы одна тема не занимала несколько мест.',
  },
  {
    title: 'Проверка кандидатов',
    text: 'Проверяется, является ли название конкретной технологией и относится ли оно к запросу. На этом этапе система ещё не решает, слабый это сигнал или нет.',
  },
  {
    title: 'Обогащение',
    text: 'Для каждого подходящего кандидата отдельно собираются научные и отраслевые документы в историческом и недавнем временных окнах.',
  },
  {
    title: 'Оценка модели',
    text: 'Интерпретируемая логистическая регрессия сравнивает кандидата с размеченными слабыми сигналами, зрелыми технологиями и рекламным шумом.',
  },
  {
    title: 'Проверка доказательств',
    text: 'Дословные фрагменты сверяются с источниками. Учитываются независимость публикаций, признаки новизны, роста, внедрения, зрелости и рекламных обещаний.',
  },
  {
    title: 'Итоговая выдача',
    text: 'Темы распределяются в основную выдачу, наблюдение и исключения. TOP-15 строится только внутри текущего пользовательского запроса.',
  },
]

const featureRows = [
  ['Динамика исследований', 'Число публикаций в предыдущем и недавнем окнах, изменение и доля документов по теме.'],
  ['Отраслевой интерес', 'Наличие свежих отраслевых публикаций и подтверждений из веб-источников.'],
  ['Качество покрытия', 'Полнота поиска, документы без даты и успешный нулевой результат отделяются от ошибки источника.'],
  ['Независимость', 'Число независимых первоисточников, организаций, издателей и типов источников.'],
  ['Надёжность', 'Наличие источников уровней доверия A/B и содержательных дословных фрагментов.'],
  ['Содержание темы', 'Слова и устойчивые сочетания в названии и проверенных вариантах названия.'],
]

const statusRows = [
  ['Основная выдача', 'Модель видит признаки слабого сигнала, проверка доказательств завершена, а независимых подтверждений достаточно.'],
  ['Требуют наблюдения', 'Сигнал возможен, но пока не хватает покрытия, независимых источников или надёжных доказательств.'],
  ['Исключены', 'Тема зрелая, преимущественно рекламная, повторная, нерелевантная либо не прошла обязательную проверку.'],
]

const metrics = [
  ['Accuracy', '91,5%'],
  ['Precision', '89,1%'],
  ['Recall', '98,0%'],
  ['F1', '93,3%'],
  ['Balanced accuracy', '89,6%'],
  ['Specificity', '81,3%'],
]

export function MethodologyPage() {
  return (
    <article className="mx-auto max-w-4xl px-4 py-10 sm:px-8 sm:py-14">
      <div className="flex size-10 items-center justify-center rounded-full bg-foreground text-background">
        <BookOpen className="size-4.5" />
      </div>
      <p className="mt-5 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">О методе</p>
      <h1 className="mt-2 text-4xl font-semibold tracking-[-0.045em]">Как NextWave ищет слабые сигналы</h1>
      <p className="mt-4 max-w-3xl text-pretty text-base leading-relaxed text-muted-foreground">
        Слабый сигнал — это ранняя, конкретная технологическая тема, у которой уже появились проверяемые
        признаки развития, но которая ещё не стала массовой и зрелой. Одного упоминания, рекламного обещания
        или общего понятия для этого недостаточно.
      </p>

      <Alert className="mt-6">
        <CheckCircle2 />
        <AlertTitle>Что именно выдаёт система</AlertTitle>
        <AlertDescription>
          NextWave не пытается описать весь интернет. Сервис строит воспроизводимую выборку открытых
          источников, находит технологические темы, оценивает их признаки и показывает, почему каждая тема
          попала в основную выдачу, наблюдение или исключения.
        </AlertDescription>
      </Alert>

      <Separator className="my-9" />
      <h2 className="mb-4 text-xl font-semibold">Путь от запроса до TOP-15</h2>
      <div className="grid gap-3 md:grid-cols-2">
        {stages.map((stage, index) => (
          <Card key={stage.title}>
            <CardHeader>
              <CardTitle className="flex items-center gap-3 text-base">
                <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground text-xs text-background">
                  {index + 1}
                </span>
                {stage.title}
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm leading-relaxed text-muted-foreground">{stage.text}</CardContent>
          </Card>
        ))}
      </div>

      <Separator className="my-9" />
      <h2 className="mb-2 text-xl font-semibold">Временные окна</h2>
      <p className="mb-4 text-sm leading-relaxed text-muted-foreground">
        Анализ зафиксирован на 15 сентября 2026 года. Предыдущее окно: с 15 сентября 2024 года включительно
        до 15 сентября 2025 года. Недавнее окно: с 15 сентября 2025 года включительно до 15 сентября 2026
        года включительно. Документы без надёжной даты сохраняются для качественного разбора, но не
        подменяют временную динамику.
      </p>

      <Separator className="my-9" />
      <h2 className="mb-4 text-xl font-semibold">Какие признаки использует модель</h2>
      <Table>
        <TableHeader><TableRow><TableHead className="w-52">Группа признаков</TableHead><TableHead>Что измеряется</TableHead></TableRow></TableHeader>
        <TableBody>
          {featureRows.map(([name, description]) => (
            <TableRow key={name}><TableCell className="font-medium">{name}</TableCell><TableCell className="text-muted-foreground">{description}</TableCell></TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
        Модель возвращает сравнительный балл от 0 до 1. В интерфейсе он показан как оценка из 100 для
        удобства чтения. Это не вероятность и не процент уверенности. Внутренний порог 0,5 нужен для
        бинарного решения модели, но финальный статус дополнительно зависит от доказательств. Поэтому две
        темы с одинаковым набором признаков могут иметь одинаковую оценку, но попасть в разные категории.
      </p>

      <Separator className="my-9" />
      <h2 className="mb-4 text-xl font-semibold">Почему тема попадает в конкретную категорию</h2>
      <div className="grid gap-3">
        {statusRows.map(([name, description]) => (
          <Card key={name}><CardHeader><CardTitle className="text-base">{name}</CardTitle></CardHeader><CardContent className="text-sm text-muted-foreground">{description}</CardContent></Card>
        ))}
      </div>

      <Separator className="my-9" />
      <h2 className="mb-2 text-xl font-semibold">Интерпретируемость</h2>
      <p className="text-sm leading-relaxed text-muted-foreground">
        Для каждой темы сохраняются ведущие положительные и отрицательные вклады признаков. Их сумма вместе
        со свободным членом воспроизводит оценку модели. Отдельно показываются дословные подтверждающие и
        опровергающие фрагменты, ссылка, дата, тип и уровень доверия источника. Решение категории формируется
        прозрачными правилами поверх оценки модели: завершённость проверки, независимость источников,
        зрелость, рекламный характер и наличие надёжного подтверждения.
      </p>

      <Separator className="my-9" />
      <h2 className="mb-2 text-xl font-semibold">Проверка качества модели</h2>
      <p className="mb-4 text-sm leading-relaxed text-muted-foreground">
        Диагностическая оценка выполнена методом детерминированной групповой 5-fold cross-validation:
        варианты одной технологии не разделяются между обучением и проверкой. Использовано 100 подтверждённых
        слабых сигналов и 64 проверенных отрицательных примера.
      </p>
      <div className="grid gap-3 sm:grid-cols-3">
        {metrics.map(([name, value]) => <Card key={name}><CardContent><div className="text-2xl font-semibold tabular-nums">{value}</div><div className="mt-1 text-xs text-muted-foreground">{name}</div></CardContent></Card>)}
      </div>
      <Alert className="mt-5">
        <BookOpen />
        <AlertTitle>Как интерпретировать метрики</AlertTitle>
        <AlertDescription>
          Система выполняет полный цикл поиска, оценки и проверки источников. Показанные метрики получены на
          зафиксированной контрольной выборке и воспроизводятся автоматически. Класс рекламного шума в ней
          представлен слабее зрелых технологий, поэтому качество распознавания этого частного случая требует
          дальнейшей проверки на расширенной выборке. Это ограничение оценки, а не ограничение работы продукта.
        </AlertDescription>
      </Alert>
      <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
        В ТЗ отдельно приветствуется число сигналов с уверенностью выше 75%. Сейчас этот счётчик намеренно
        не показывается: модельный балл ещё не откалиброван как вероятность. Подменять его «уверенностью»
        было бы методологически неверно.
      </p>

      <Separator className="my-9" />
      <h2 className="mb-3 text-xl font-semibold">Что можно проверить на стенде</h2>
      <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-muted-foreground">
        <li>Свободный запрос запускает поиск вне заранее заданного перечня технологий.</li>
        <li>В результате видны число проверенных названий, найденных тем и обработанных источников.</li>
        <li>Для каждой темы доступны оценка модели, ключевые предикторы, краткое русское резюме, преимущество и кейс-пример.</li>
        <li>Зарубежные источники сохраняют оригинальные реквизиты, а автоматическая русская интерпретация помечена.</li>
        <li>Наблюдение и исключения доступны вместе с конкретной причиной решения.</li>
        <li>Отчёт о модели показывает Precision, Recall, F1 и Accuracy выше требуемых 75–80% как диагностические метрики.</li>
      </ul>

      <Separator className="my-9" />
      <h2 className="mb-3 text-xl font-semibold">Ограничения и воспроизводимость</h2>
      <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-muted-foreground">
        <li>Поиск ограничен бюджетом страниц, документов и времени; отсутствие темы в выдаче не доказывает её отсутствие в мире.</li>
        <li>Успешный поиск без результатов, ошибка источника и пропущенный запрос учитываются по-разному.</li>
        <li>Дубликаты и перепечатки не считаются независимыми подтверждениями.</li>
        <li>Первое найденное упоминание — дата в собранном корпусе, а не гарантированная дата появления технологии.</li>
        <li>Для анализа сохраняются запрос, дата отсечения, версии метода, модели и контрольные суммы результатов.</li>
        <li>TOP-15 — приоритетный список для аналитика, а не обещание коммерческого успеха технологии.</li>
      </ul>
    </article>
  )
}
