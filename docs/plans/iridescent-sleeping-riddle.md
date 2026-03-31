# Рефакторинг Docker: только внешние зависимости в образе

## Контекст

Dockerfile зашивает код проекта (src/) и модели (models/) в образ при сборке. При mount `./src:/app/src` это ломает detectron2 (develop mode) и удалённые файлы. Каждое изменение кода требует пересборки. Нужно разделить: образ = внешние зависимости, runtime = код из volume + модели из volume.

## План (9 файлов, 0 новых)

### 1. `Dockerfile` — убрать всё кроме внешних зависимостей

- Объединить 16 `RUN apt-get install` в один слой с `apt-get clean`
- Клонировать detectron2 в `/tmp/detectron2` вместо `/app/src/detectron2`
- Удалить: `COPY ./src/download_models.py`, `COPY ./src/configuration.py`, `COPY ./models/`, `RUN python src/download_models.py`, `RUN rm`
- Добавить: `ENTRYPOINT ["./start.sh"]`
- Оставить: requirements.txt install, detectron2 install, pycocotools, start.sh copy

### 2. `start.sh` — скачивание моделей при старте

```bash
#!/bin/bash
set -e
PYTHONPATH=/app/src python -c "
from download_models import download_models
download_models('doclaynet')
download_models('fast')
"
exec gunicorn -k uvicorn.workers.UvicornWorker --chdir ./src app:app --bind 0.0.0.0:5060 --timeout 10000
```

- `set -e` — падаем при ошибке скачивания
- `exec` — gunicorn получает PID 1, корректный SIGTERM
- Идемпотентно: download_models проверяет exists() перед скачиванием

### 3. `.dockerignore` — исключить src/ и models/ из контекста билда

Добавить: `/src/`, `/models/`, `*.pdf`, `.idea/`, `__pycache__/`

### 4-9. Все 6 docker-compose файлов

Для каждого сервиса `pdf-document-layout-analysis*`:
- **Удалить** `entrypoint:` (используется ENTRYPOINT из Dockerfile → start.sh)
- **Добавить** `- ./models:/app/models` в volumes
- `docker-compose.yml`: добавить volumes секцию с `./src:/app/src` и `./models:/app/models`

Файлы: docker-compose.yml, docker-compose-gpu.yml, docker-compose-ocr-openrouter.yml, docker-compose-ocr-ollama.yml, docker-compose-ocr-llama-cpp.yml, docker-compose-ocr-vllm.yml

GUI сервисы (Dockerfile.gradio) — не трогаем.

## Порядок выполнения

1. Dockerfile + start.sh + .dockerignore
2. docker-compose-ocr-openrouter.yml (целевой для теста)
3. `docker compose -f docker-compose-ocr-openrouter.yml build`
4. `docker compose -f docker-compose-ocr-openrouter.yml up` — проверить что модели скачиваются и сервис стартует
5. Тест: отправить page_121.pdf на `/markdown` endpoint
6. Остальные 5 compose файлов

## Персистентность моделей

- Volume `./models:/app/models` — модели хранятся на хосте, переживают `docker compose down` и пересоздание контейнера
- `download_models.py` проверяет `exists()` для каждого файла/директории перед скачиванием — повторная загрузка не происходит
- При повторном `docker compose up` проверка моделей занимает миллисекунды

## Верификация

1. `docker compose -f docker-compose-ocr-openrouter.yml up --build` — стартует без ошибок
2. Логи показывают скачивание моделей при первом запуске
3. `curl -X POST http://localhost:5060/markdown -F "file=@page_121.pdf"` — возвращает markdown
4. `docker compose down && docker compose up` — модели НЕ скачиваются повторно (логи: "already exists")
5. `ls ./models/` на хосте — файлы моделей присутствуют после остановки контейнера
