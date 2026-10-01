"""
Mental Manager — веб-backend.
Один процесс FastAPI: отдаёт сайт (static/index.html) и эндпоинт /api/chat.

Вся логика бота («мозг») живёт в общем модуле mental_core.py — том же самом,
что использует Telegram-бот. Сайт ничего не дублирует: он просто вызывает
mental_core.process_message(), то есть на сайте отвечает ровно тот же бот
(тот же промпт «Ментал», кризис-детектор, база знаний, память, ключи из .env).
"""

import sys
from pathlib import Path

# Чтобы импортировать mental_core.py и storage.py из корня проекта
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import mental_core  # noqa: E402  — общий «мозг» бота

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Mental Manager Web")


@app.middleware("http")
async def no_cache(request, call_next):
    # Локальный сервер всегда отдаёт свежую версию — чтобы на показе не всплыл старый кэш
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


class ChatIn(BaseModel):
    message: str
    session_id: str = "web-default"
    user_name: str = ""


class ChatOut(BaseModel):
    reply: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "yandex_configured": bool(mental_core.YANDEX_API_KEY and mental_core.YANDEX_FOLDER_ID),
        "ml_available": mental_core.embedder is not None,
        "kb_entries": len(mental_core.kb_texts),
    }


@app.get("/api/selftest")
async def selftest():
    """Живая проверка ключей: делает реальный минимальный запрос к YandexGPT."""
    reply = await mental_core.get_yandex_gpt_response(
        "Ответь ровно одним словом: работает.", "проверка связи"
    )
    ok = bool(reply) and not reply.startswith(("Давай", "Расскажи"))
    return {"yandex_ok": ok, "reply": reply}


@app.post("/api/chat", response_model=ChatOut)
async def chat(inp: ChatIn):
    reply = await mental_core.process_message(inp.session_id, inp.message)
    return ChatOut(reply=reply)


@app.post("/api/reset")
def reset(inp: ChatIn):
    # Начать чат заново — забываем профиль этой веб-сессии
    mental_core.user_profiles.pop(str(inp.session_id), None)
    mental_core.storage.save_profiles(mental_core.user_profiles)
    return {"status": "reset"}


# Отдаём сайт. Монтируем ПОСЛЕ api-роутов, чтобы /api/* не перехватывался.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="site")


if __name__ == "__main__":
    import uvicorn

    print("=" * 55)
    print("  Mental Manager - web server")
    print(f"  YandexGPT: {bool(mental_core.YANDEX_API_KEY and mental_core.YANDEX_FOLDER_ID)}")
    print(f"  ML (semantic): {mental_core.embedder is not None}")
    print("  Open in browser:  http://localhost:8000")
    print("=" * 55)
    uvicorn.run(app, host="127.0.0.1", port=8000)
