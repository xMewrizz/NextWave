# V1-10: интеграция финальных оценок

Backend больше не использует `demo_trends.json`: без готового анализатора запуск получает
`status=error`, `error_code=model_unavailable` и сообщение «Интеграция модели недоступна».
Этот MR подготавливает интеграцию, но не реализует V1-08/V1-09 и не доказывает готовность
полного live-поиска. Discovery, Candidate Gate, feature builder и обучение не заменяются.

## Что передать ML-разработчику

### Точка подключения

Общий, независимый от backend/UI протокол: `src/nextwave/analysis.py::Analyzer`.
Настройка сервера: `NEXTWAVE_ANALYZER_FACTORY=nextwave.ml.integration:build_analyzer`.
Имя приведено как пример: этот модуль должен предоставить ML-разработчик.
Фабрика без аргументов создаёт отдельный экземпляр анализатора для каждого запуска:

```python
from nextwave.analysis import AnalysisMetadata
from nextwave.contracts import CandidateAssessment

class RealAnalyzer:
    # Подставить реальные версии артефактов и правил, а не строки из test fixtures.
    metadata: AnalysisMetadata

    def analyze(self, query: str) -> list[CandidateAssessment]:
        # Здесь вызывается существующий ML-конвейер:
        # discovery → проверка evidence → общий feature builder → модель → decision policy.
        # Вернуть ВСЕ финальные оценки; ранний Gate reject не является финальным статусом.
        raise NotImplementedError("Подключить вычислительный код ML")


def build_analyzer() -> RealAnalyzer:
    # Загрузить реальные артефакты и установить metadata до возврата экземпляра.
    raise NotImplementedError("Подключить загрузку артефактов ML")
```

Фабрика и синхронный `analyze(query)` выполняются в рабочем потоке, вне event loop API.
Возвращается `list` или `tuple` объектов `CandidateAssessment` либо соответствующих словарей.
Генераторы, coroutine, `DiscoveryPipelineResult` и JSON-строки не принимаются.
Новый JSON-конверт результата не вводится. Динамический импорт определяется только конфигурацией
сервера, никогда пользовательским запросом. В импортируемом модуле не должно быть backend/UI-зависимостей.

`metadata` — `AnalysisMetadata` или эквивалентный словарь:

```json
{
  "method_version": "contract-test-policy-v1",
  "model_version": "contract-test-model-v1",
  "feature_version": "contract-test-features-v1",
  "corpus_version": "contract-test-corpus-v1"
}
```

Это учебные версии. В production передать версию decision policy, точную версию модели,
feature builder и идентификатор корпуса/snapshot, на котором выполняется запуск.
Фабрика должна зарезервировать идентификатор snapshot до вычисления. Версии непустые;
`method_version` и `corpus_version` ограничены 100 символами существующей схемой БД.
Метаданные сохраняются до `analyze`, поэтому пустой результат и ошибка вычисления воспроизводимо
относятся к выбранным версиям. До загрузки фабрики версии неизвестны: nullable-поля равны `null`,
существующие обязательные `corpus_version`/`method_version` равны `unavailable`.

### Точный контракт

Источник истины — Pydantic `CandidateAssessment` в `src/nextwave/contracts.py`.
TypeScript генерируется из тех же моделей, отдельного ML-формата нет.

| Поле | Что передать |
| --- | --- |
| `candidate_id` | Уникальный внутри запуска непустой идентификатор |
| `canonical_name` | Непустое каноническое название |
| `query` | Точный исходный запрос после удаления крайних пробелов, переданный в `analyze` |
| `features` | `CandidateFeatures`; неизвестные nullable-признаки остаются `null` |
| `prediction.weak_signal_score` | Реальная оценка модели в `[0, 1]` |
| `prediction.model_version`, `prediction.feature_version` | Совпадают с metadata запуска |
| `prediction.calibrated` | По умолчанию `false`; оценку не называем вероятностью |
| `status` | Только `main`, `watchlist`, `excluded` после decision policy |
| `explanation` | Непустое объяснение решения по реальным признакам и evidence |
| `exclusion_reason` | Обязателен только для `excluded`, иначе `null` |
| `evidence` | Массив `Evidence`; для `main` минимум один элемент |
| `priority_score` | Существующее необязательное поле `[0, 100]`; сохраняется, но порядок ТОП задаёт модель |

Обязательные признаки: `stage`, `independent_source_count`, `source_type_diversity`,
`independent_actor_count`. Остальные поля текущего `CandidateFeatures` nullable:
`publication_momentum`, `patent_momentum`, `evidence_recency_days`, `mass_adoption`,
`formed_market`, `industry_standard`, `promotional_source_share`.
Новый feature builder должен согласовать реальную схему с этим контрактом; адаптер не
придумывает недостающие признаки и не принимает неизвестные поля.

Допустимые причины исключения: `mature`, `mass_adoption`, `industry_standard`,
`marketing_hype`, `insufficient_trust`, `irrelevant`, `duplicate`.

В `Evidence` обязательны `evidence_id`, `title`, абсолютный HTTP(S) `url`, `source_type`,
`language`, `trust_level`. `published_at` — дата публикации либо `null`;
`retrieved_at` — время получения; `excerpt` — исходная цитата; `generated_summary` отмечает
генеративное резюме. Минимальное совместимое дополнение — `direction: support | counter | null`.
Старые записи без направления читаются с `null`; UI честно сообщает, что оно не указано.

Из проверенного `EvidenceProposal` сохранить `claim.text` в `Evidence.excerpt`,
`claim.direction` в `Evidence.direction`, `claim.claim_id` в `evidence_id`,
`source_url` в `url`; название, дату, тип и язык брать из связанного документа.
Предложения `pending` не становятся подтверждёнными evidence автоматически.
Уровень `TrustTier` документа и `trust_level` финального контракта — разные шкалы;
их соответствие и допуск к решению обеспечивает ML/decision policy, адаптер их не угадывает.
Восстановлены существующие discovery-типы `ClaimType`, `EvidenceDirection`, `EvidenceClaim`
и типы источников, удалённые при предыдущей унификации: discovery снова импортируется.

В payload не передаются API-ключи, секреты, полные сырые ответы или персональные данные.
Произвольные поля запрещены контрактом; за содержание допустимых текстовых полей отвечает
анализатор. Исходный текст исключений не попадает в API и БД.

### Примеры входа и выхода

Вход HTTP: `POST /api/analyses` с `{"query":"Проверка интеграции"}`.
Вход ML: `analyze("Проверка интеграции")`.

- [Полный валидный выход ML](examples/ml-assessments.json).
- [Полный JSON ответа GET API](examples/api-analysis.json).

Оба файла — явно учебные test fixtures, а не реальные модельные выводы. Production их не читает.
Они проверяются backend-тестами. Когда готов ML, нужно заменить учебные значения фактическими
выходами его вычисления; пример API показывает точную структуру, включая `null` вместо
отсутствующих описаний, кейса, количества документов и первой даты.

### Локальная проверка

Из корня репозитория:

```bash
uv run --project backend pytest backend/test_api.py -q
uv run --project backend python backend/generate_frontend_contracts.py --check
```

Тесты подставляют анализатор явно и проверяют все три статуса, ошибки, пустой результат,
сохранение исходных полей, неблокирующий запуск и повторный GET без ML.
Для реального PostgreSQL (тестовый пользователь должен иметь право CREATE SCHEMA):

```bash
NEXTWAVE_TEST_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@localhost:5432/TEST_DB' \
  uv run --project backend pytest backend/test_api.py -q
```

Каждый тест создаёт и удаляет только свою случайную схему `test_<uuid>`;
пользовательские таблицы не затрагиваются. Без переменной используется временная SQLite.

После реализации фабрики запустить backend, передав реальные настройки:

```bash
NEXTWAVE_ANALYZER_FACTORY=nextwave.ml.integration:build_analyzer \
NEXTWAVE_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@localhost:5432/nextwave' \
  uv run --project backend uvicorn app.main:app --app-dir backend --port 8000
curl -X POST http://localhost:8000/api/analyses \
  -H 'Content-Type: application/json' -d '{"query":"технологии в ИИ"}'
curl http://localhost:8000/api/analyses/ID_ИЗ_POST
```

Для UI: `cd frontend && npm run dev`. В Docker фабрика должна входить в установленный
Python-пакет; затем `NEXTWAVE_ANALYZER_FACTORY=nextwave.ml.integration:build_analyzer docker compose up --build`.
От ML ещё ожидаются вычислительный конвейер, артефакт модели, feature builder, decision policy,
проверенные evidence и реальные версии. Этот MR не вычисляет их за ML.

## API, хранение и ограничения

Существующие маршруты сохранены: POST и список `/api/analyses`, GET `/api/analyses/{id}`,
GET `/api/analyses/{id}/trends/{candidate_id}`, `/api/stages`, `/api/coverage`.

`pending` создаётся до запуска задачи; `running` показывает фактическую фазу, а не таймер
с вымышленными этапами. Во время вычисления процент остаётся 0; 80% и 95% обозначают
валидацию и подготовку сохранения, а не измеренную долю выполненных ML-операций.
`done` означает непустой валидный результат, даже если в нём нет `main`;
`empty` — успешно полученный пустой список. Ошибки: `model_unavailable`,
`invalid_model_result`, `analysis_failed`, `interrupted`. Финальные состояния имеют дату завершения.

Адаптер не переклассифицирует кандидатов. Все финальные оценки сохраняются в существующем
`analyses.trends` (JSON), вместе с признаками, prediction и evidence. Backend добавляет только
`rank` внутри статуса: model score по убыванию, число независимых источников по убыванию,
затем стабильный ID. UI показывает `main` с `rank <= 15`; если их больше, API хранит всех
и сообщает об ограничении отображения. `watchlist`/`excluded` показываются полностью.
Результат с меньшим числом main не дополняется искусственными кандидатами.

`Trend` остаётся расширением `CandidateAssessment`. Старые презентационные поля сохранены,
но теперь необязательны; отсутствующий кейс, график, summary или счётчик не фабрикуется.
На карточке всегда видны реальные признаки, score, версии, explanation и доступные evidence.
`/coverage` возвращает неизвестную статистику (`null`, пустые списки, объяснение), поскольку
пока нет реального агрегатора покрытия; старые демо-числа и пороги удалены.

`init_database()` создаёт таблицу на чистой БД либо идемпотентно добавляет nullable-колонки
`started_at`, `model_version`, `feature_version`, `error_code` к старой `analyses`.
В PostgreSQL DDL выполняется транзакционно под advisory lock; старые JSON не переписываются.
Старая выдача с `demo-`/`synthetic-` обозначается как демонстрационная. Новые GET читают только БД.
Перезапуск помечает незавершённые задачи как `error/interrupted`; автоматического повторного ML нет.

MVP запускается в одном процессе uvicorn. In-process очередь не предназначена для нескольких
workers/реплик: startup recovery иначе может прервать чужую задачу. Для масштабирования нужна
отдельная очередь с владением задачами; это не добавлено в V1-10. Отмена async-задачи не останавливает
синхронный поток ML, поэтому анализатор обязан ограничивать время внешних запросов и вычисления.

## Ручная проверка интерфейса

1. Без фабрики отправить запрос: получить явную ошибку недоступной интеграции без карточек.
2. С реальным анализатором проверить загрузку, ТОП с фактическим числом и вкладки наблюдения/исключения.
3. Открыть карточку: проверить score, версии, неизвестные признаки, цитату, дату, ссылку и «За/Против».
4. Обновить страницу после завершения: выдача сохраняется, ML не вызывается снова. Проверить узкую ширину.
