# Docker-запуск NextWave

## API, интерфейс и PostgreSQL

Compose запускает API, web-интерфейс и PostgreSQL 16 с постоянным volume `postgres_data`.
Таблица `analyses` создаётся или совместимо обновляется при старте backend.
Без ML-фабрики API доступен, но запуск анализа возвращает `error/model_unavailable`.
Подключение реального вычислительного кода описано в [ML_INTEGRATION.md](docs/ML_INTEGRATION.md).
Для локального стенда задан пароль `nextwave-local`; его можно переопределить через
`NEXTWAVE_POSTGRES_PASSWORD` (URL-safe значение). PostgreSQL не публикует порт наружу.

Из корня репозитория:

```bash
docker compose up --build
```

Интерфейс будет доступен на `http://localhost:8080`, API — на `http://localhost:8000`.

Compose автоматически передаёт backend временный Media Cloud token из `config/hackathon.env`; регистрация и ручной ввод ключа для проверки приватного хакатонного репозитория не требуются. Переменная недоступна frontend-контейнеру. После завершения оценки token отзывается.
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
  -e NEXTWAVE_DATABASE_URL -e NEXTWAVE_ANALYZER_FACTORY nextwave-backend
```

Frontend при отдельном развёртывании должен знать адрес backend. В PowerShell:

```powershell
docker build -t nextwave-frontend ./frontend
docker run --rm -p 8080:80 -e BACKEND_ORIGIN=http://host.docker.internal:8000 nextwave-frontend
```

Frontend обслуживается nginx и проксирует запросы `/api/*` в `BACKEND_ORIGIN`.
