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

from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import mental_core  # noqa: E402  — общий «мозг» бота
import platform_store  # noqa: E402  — аккаунты, роли, пользовательские тесты (Этап 3)

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
    name: str = ""
    faculty: str = ""
    course: str = ""


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
    student = {"name": inp.name or inp.user_name, "faculty": inp.faculty, "course": inp.course}
    reply = await mental_core.process_message(inp.session_id, inp.message, student=student)
    return ChatOut(reply=reply)


@app.post("/api/reset")
def reset(inp: ChatIn):
    # Начать чат заново — забываем профиль этой веб-сессии
    mental_core.user_profiles.pop(str(inp.session_id), None)
    mental_core.storage.save_profiles(mental_core.user_profiles)
    return {"status": "reset"}


# ====== Платформа: аккаунты, роли, пользовательские тесты (Этап 3) ======
class SignupIn(BaseModel):
    username: str
    password: str
    role: str
    name: str = ""
    admin_code: str = ""


class LoginIn(BaseModel):
    username: str
    password: str


class TestIn(BaseModel):
    title: str = ""
    intro: str = ""
    instruction: str = ""
    options: list = []
    questions: list = []
    bands: list = []
    entry_fields: list = []
    published: bool = True


class SubmitIn(BaseModel):
    test_id: str
    entry: dict = {}
    answers: list = []
    result: dict = {}


def _as_dict(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _token(authorization):
    if not authorization:
        return ""
    return authorization.split()[-1].strip()


def _require(authorization):
    me = platform_store.whoami(_token(authorization))
    if not me:
        raise HTTPException(status_code=401, detail="Требуется вход")
    return me


@app.post("/api/platform/signup")
def platform_signup(inp: SignupIn):
    me, err = platform_store.create_account(inp.username, inp.password, inp.role, inp.name, inp.admin_code)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return me


@app.post("/api/platform/login")
def platform_login(inp: LoginIn):
    me, err = platform_store.login(inp.username, inp.password)
    if err:
        raise HTTPException(status_code=401, detail=err)
    return me


@app.get("/api/platform/me")
def platform_me(authorization: str = Header(None)):
    return platform_store.whoami(_token(authorization)) or {}


@app.post("/api/platform/logout")
def platform_logout(authorization: str = Header(None)):
    platform_store.logout(_token(authorization))
    return {"status": "ok"}


@app.post("/api/platform/tests")
def platform_create_test(inp: TestIn, authorization: str = Header(None)):
    me = _require(authorization)
    return platform_store.create_test(me["username"], _as_dict(inp))


@app.get("/api/platform/tests/mine")
def platform_my_tests(authorization: str = Header(None)):
    me = _require(authorization)
    return platform_store.list_tests(owner=me["username"])


@app.get("/api/platform/tests/published")
def platform_published_tests():
    return platform_store.list_tests(published=True)


@app.get("/api/platform/tests/{tid}")
def platform_get_test(tid: str):
    t = platform_store.get_test(tid)
    if not t:
        raise HTTPException(status_code=404, detail="Тест не найден")
    return t


@app.delete("/api/platform/tests/{tid}")
def platform_delete_test(tid: str, authorization: str = Header(None)):
    me = _require(authorization)
    if not platform_store.delete_test(me["username"], tid):
        raise HTTPException(status_code=404, detail="Тест не найден или принадлежит не вам")
    return {"status": "deleted"}


@app.post("/api/platform/submit")
def platform_submit(inp: SubmitIn):
    if not platform_store.get_test(inp.test_id):
        raise HTTPException(status_code=404, detail="Тест не найден")
    rec = platform_store.add_submission(_as_dict(inp))
    return {"status": "ok", "id": rec["id"]}


@app.get("/api/platform/submissions")
def platform_submissions(authorization: str = Header(None)):
    me = _require(authorization)
    return platform_store.list_submissions(me["username"])


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
