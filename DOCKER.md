# Docker-запуск NextWave

## API, интерфейс и PostgreSQL

Compose запускает API, web-интерфейс и PostgreSQL 16 с постоянным volume `nextwave-postgres`.
Backend хранит analysis jobs в PostgreSQL. По умолчанию он выдаёт проверенный snapshot
для совпадающего запроса. Для произвольных запросов установите `NEXTWAVE_ANALYSIS_MODE=live`;
настройка и необходимые ключи описаны в [README.md](README.md#запуск-проверенного-web-результата).
Для локального стенда задан пароль `nextwave-local`; его можно переопределить через
`NEXTWAVE_POSTGRES_PASSWORD` (URL-safe значение). PostgreSQL не публикует порт наружу.

Из корня репозитория:

```bash
docker compose up --build
```

Интерфейс будет доступен на `http://localhost:8080`, API — на `http://localhost:8000`.

Compose передаёт backend настройки из `config/hackathon.env`; секреты для live-анализа
задаются переменными окружения и не попадают во frontend.
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
