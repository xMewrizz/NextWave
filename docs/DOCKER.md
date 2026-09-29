# Docker-запуск NextWave

## Проверенный демонстрационный результат

По умолчанию Compose запускает PostgreSQL, FastAPI и React и показывает
зафиксированный результат реального открытого поиска из `demo/result`. Bundle
проверяется по SHA-256. Внешние API и LLM при таком запуске не вызываются.

Из корня репозитория:

```bash
docker compose up --build
```

Интерфейс будет доступен на `http://localhost:8080`, API — на `http://localhost:8000`.

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

## Раздельная сборка

Backend:

```bash
docker build -t nextwave-backend ./backend
docker run --rm -p 8000:8000 nextwave-backend
```

Frontend при отдельном развёртывании должен знать адрес backend. В PowerShell:

```powershell
docker build -t nextwave-frontend ./frontend
docker run --rm -p 8080:80 -e BACKEND_ORIGIN=http://host.docker.internal:8000 nextwave-frontend
```

Frontend обслуживается nginx и проксирует запросы `/api/*` в `BACKEND_ORIGIN`.

## Произвольный живой запрос

Для live-режима создайте локальный `.env` по `.env.example`, задайте серверные
`NEXTWAVE_OPENAI_API_KEY`, `NEXTWAVE_EXA_API_KEY` и при наличии
`NEXTWAVE_OPENALEX_API_KEY`, затем выполните:

```powershell
$env:NEXTWAVE_ANALYSIS_MODE = "live"
docker compose up --build -d
```

Ключи получает только backend. Они не включаются во frontend, snapshots,
manifest или Git.
