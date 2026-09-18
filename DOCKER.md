# Docker-запуск NextWave

## Демонстрационный каркас

Текущая конфигурация запускает синтетический UI-каркас и API. Она проверяет сборку и пользовательский сценарий, но не содержит реальных коннекторов, модели и PostgreSQL.

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
