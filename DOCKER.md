# Docker-запуск NextWave

## Весь проект

Из корня репозитория:

```bash
docker compose up --build
```

Интерфейс будет доступен на `http://localhost:8080`, API — на `http://localhost:8000`.
Порты можно изменить переменными `FRONTEND_PORT` и `BACKEND_PORT`.

```bash
FRONTEND_PORT=3000 BACKEND_PORT=8001 docker compose up --build
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

Frontend при отдельном развёртывании должен знать адрес backend:

```bash
docker build -t nextwave-frontend ./frontend
docker run --rm -p 8080:80 \
  -e BACKEND_ORIGIN=http://host.docker.internal:8000 \
  nextwave-frontend
```

Frontend обслуживается nginx и проксирует запросы `/api/*` в `BACKEND_ORIGIN`.
