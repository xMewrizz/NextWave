# Docker-запуск NextWave

## Проверенный демонстрационный результат

По умолчанию Compose запускает PostgreSQL, FastAPI и React и показывает
зафиксированный результат реального открытого поиска из `demo/result`. Bundle
проверяется по SHA-256. Внешние API и LLM при таком запуске не вызываются
(режим `cached_snapshot`). Сервис отвечает только на запрос этого snapshot,
«Инфраструктурные технологии для обучения и инференса ИИ»; любой другой запрос
завершается понятной ошибкой job и не получает подставленного результата.

## API, интерфейс и PostgreSQL

Compose запускает API, web-интерфейс и PostgreSQL 16 с постоянным volume `nextwave-postgres`.
Backend хранит analysis jobs в PostgreSQL, поэтому они переживают `docker compose restart backend`.
Для произвольных запросов включите живой режим (см. ниже).
Для локального стенда задан пароль `nextwave-local`; его можно переопределить через
`NEXTWAVE_POSTGRES_PASSWORD` (URL-safe значение). PostgreSQL не публикует порт наружу.

Из корня репозитория:

```bash
docker compose up --build
```

Интерфейс будет доступен на `http://localhost:8080`, API — на `http://localhost:8000`.

Compose передаёт backend настройки из `config/hackathon.env`; секреты для live-анализа
задаются в корневом `.env` (Compose читает его автоматически, шаблон — `.env.example`)
или переменными окружения и не попадают во frontend.
В PowerShell порты можно изменить переменными `FRONTEND_PORT` и `BACKEND_PORT`:

```powershell
$env:FRONTEND_PORT = "3000"
$env:BACKEND_PORT = "8001"
docker compose up --build
```

Остановка:

```bash
docker compose down
```

Обычный `docker compose down` сохраняет volume базы; `down -v` удаляет данные и не нужен для обновления.

## Раздельная сборка

Backend:

```bash
docker build -f backend/Dockerfile -t nextwave-backend .
docker run --rm -p 8000:8000 \
  -e NEXTWAVE_DATABASE_URL -e NEXTWAVE_ANALYSIS_MODE \
  -e NEXTWAVE_RESULT_DIR=/data/result -v /path/to/analysis-result-v2:/data/result:ro \
  nextwave-backend
```

Frontend при отдельном развёртывании должен знать адрес backend. В PowerShell:

```powershell
docker build -t nextwave-frontend ./frontend
docker run --rm -p 8080:80 -e BACKEND_ORIGIN=http://host.docker.internal:8000 nextwave-frontend
```

Frontend обслуживается nginx и проксирует запросы `/api/*` в `BACKEND_ORIGIN`.

## Произвольный живой запрос

Живой режим тратит платные квоты, поэтому включается только явно. Создайте
`.env` по `.env.example`, задайте серверные ключи и режим:

```bash
cp .env.example .env
# заполните NEXTWAVE_OPENAI_API_KEY и NEXTWAVE_EXA_API_KEY, NEXTWAVE_ANALYSIS_MODE=live
docker compose up --build -d
```

Обязательны `NEXTWAVE_OPENAI_API_KEY` и `NEXTWAVE_EXA_API_KEY`;
`NEXTWAVE_OPENALEX_API_KEY` и `NEXTWAVE_MEDIACLOUD_API_KEY` необязательны.
Без обязательных ключей job завершается ошибкой с указанием ключа, а сервис продолжает работать.
Ключи получает только backend: они не попадают во frontend, snapshots, manifest или Git.
Промежуточные результаты хранятся в volume `nextwave-analysis-work`, поэтому «Повторить»
после сбоя продолжает анализ с последней завершённой стадии.

| | `cached_snapshot` (по умолчанию) | `live` |
| --- | --- | --- |
| Запрос | только зафиксированный в `demo/result` | любой (до 200 символов) |
| Внешние вызовы | нет | LLM, Exa, OpenAlex, Media Cloud/GDELT |
| Ключи | не нужны | `NEXTWAVE_OPENAI_API_KEY`, `NEXTWAVE_EXA_API_KEY` |
| Стадии в интерфейсе | «сохранено» | обновляются по ходу анализа |
| Контракт ответа | `analysis-response-v1` | тот же |

Интерфейс показывает текущий режим на главной странице и в каждом результате.

## Проверка стенда

После `docker compose up --build -d` (`cached_snapshot`):

```bash
python scripts/web_rehearsal.py --base-url http://127.0.0.1:8080   # 120 кандидатов, TOP-15, policy-статусы
docker compose restart backend
python scripts/web_rehearsal.py --base-url http://127.0.0.1:8080 --job-id <ID из первого запуска>
```
