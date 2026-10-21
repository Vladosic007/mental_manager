"""
platform_store.py — аккаунты, роли, пользовательские тесты и ответы (Этап 3).

Отдельный модуль «платформы» поверх существующего приложения: добавляет роли
(преподаватель / психолог), вход по логину-паролю, хранение созданных ими тестов
и ответов студентов. Данные лежат в JSON-файлах рядом с проектом. Пароли хранятся
только в виде соли + PBKDF2-хэша, открытые пароли нигде не сохраняются.

Это прототип: сессии (токены) живут в оперативной памяти и сбрасываются при
перезапуске сервера — после рестарта преподавателю/психологу нужно войти заново.
"""

import os
import json
import secrets
import hashlib
import hmac
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
ACCOUNTS_FILE = ROOT / "platform_accounts.json"
TESTS_FILE = ROOT / "platform_tests.json"
SUBMISSIONS_FILE = ROOT / "platform_submissions.json"

# Код, по которому можно зарегистрировать аккаунт преподавателя/психолога.
# Берётся из .env (PLATFORM_ADMIN_CODE), иначе — значение по умолчанию (поменяйте!).
ADMIN_CODE = os.getenv("PLATFORM_ADMIN_CODE", "ufu-mp-2026")

ROLES = ("teacher", "psychologist")
ROLE_RU = {"teacher": "преподаватель", "psychologist": "психолог"}


def _load(path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[platform] не удалось прочитать {path.name}: {e}")
    return default


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


accounts = _load(ACCOUNTS_FILE, {})        # {username: {role, name, salt, pwd, created}}
custom_tests = _load(TESTS_FILE, {})       # {test_id: {...}}
submissions = _load(SUBMISSIONS_FILE, {})  # {sub_id: {...}}
sessions = {}                              # token -> username (в памяти)


# ====== Пароли и аккаунты ======
def _hash_pw(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000).hex()


def create_account(username, password, role, name="", admin_code=""):
    username = (username or "").strip().lower()
    password = password or ""
    if not username or not password:
        return None, "Укажите логин и пароль"
    if len(password) < 6:
        return None, "Пароль должен быть не короче 6 символов"
    if role not in ROLES:
        return None, "Недопустимая роль"
    if not hmac.compare_digest(admin_code or "", ADMIN_CODE):
        return None, "Неверный код регистрации"
    if username in accounts:
        return None, "Такой логин уже занят"
    salt = secrets.token_hex(16)
    accounts[username] = {
        "username": username,
        "role": role,
        "name": name.strip() or username,
        "salt": salt,
        "pwd": _hash_pw(password, salt),
        "created": datetime.now().isoformat(),
    }
    _save(ACCOUNTS_FILE, accounts)
    return _public(accounts[username]), None


def login(username, password):
    username = (username or "").strip().lower()
    acc = accounts.get(username)
    # Константное по времени сравнение даже при отсутствии аккаунта
    dummy_salt = secrets.token_hex(16)
    expected = acc["pwd"] if acc else _hash_pw("x", dummy_salt)
    candidate = _hash_pw(password or "", acc["salt"] if acc else dummy_salt)
    if not acc or not hmac.compare_digest(expected, candidate):
        return None, "Неверный логин или пароль"
    token = secrets.token_urlsafe(24)
    sessions[token] = username
    me = _public(acc)
    me["token"] = token
    return me, None


def whoami(token):
    username = sessions.get(token or "")
    acc = accounts.get(username) if username else None
    return _public(acc) if acc else None


def logout(token):
    sessions.pop(token or "", None)


def _public(acc):
    if not acc:
        return None
    return {"username": acc["username"], "role": acc["role"], "name": acc["name"]}


# ====== Пользовательские тесты ======
def create_test(owner, data):
    tid = secrets.token_hex(6)
    rec = {
        "id": tid,
        "owner": owner,
        "owner_name": accounts.get(owner, {}).get("name", owner),
        "created": datetime.now().isoformat(),
        "title": (data.get("title") or "Без названия").strip(),
        "intro": (data.get("intro") or "").strip(),
        "instruction": (data.get("instruction") or "").strip(),
        "options": data.get("options") or [],       # [{label, v}]
        "questions": data.get("questions") or [],    # [{t}]
        "bands": data.get("bands") or [],            # [{max, title, analysis}]
        "entry_fields": data.get("entry_fields") or [],  # анкета на входе
        "published": bool(data.get("published", True)),
    }
    custom_tests[tid] = rec
    _save(TESTS_FILE, custom_tests)
    return rec


def list_tests(owner=None, published=None):
    out = list(custom_tests.values())
    if owner is not None:
        out = [t for t in out if t.get("owner") == owner]
    if published is not None:
        out = [t for t in out if bool(t.get("published")) == published]
    return sorted(out, key=lambda t: t.get("created", ""), reverse=True)


def get_test(tid):
    return custom_tests.get(tid)


def delete_test(owner, tid):
    t = custom_tests.get(tid)
    if t and t.get("owner") == owner:
        custom_tests.pop(tid, None)
        _save(TESTS_FILE, custom_tests)
        return True
    return False


# ====== Ответы студентов (submissions) ======
def add_submission(data):
    sid = secrets.token_hex(8)
    tid = data.get("test_id")
    t = custom_tests.get(tid, {})
    rec = {
        "id": sid,
        "created": datetime.now().isoformat(),
        "test_id": tid,
        "test_title": t.get("title", ""),
        "owner": t.get("owner", ""),
        "entry": data.get("entry") or {},     # ответы анкеты на входе
        "answers": data.get("answers") or [],
        "result": data.get("result") or {},
    }
    submissions[sid] = rec
    _save(SUBMISSIONS_FILE, submissions)
    return rec


def list_submissions(owner):
    owned = {t["id"] for t in custom_tests.values() if t.get("owner") == owner}
    out = [s for s in submissions.values() if s.get("test_id") in owned]
    return sorted(out, key=lambda s: s.get("created", ""), reverse=True)


def stats():
    return {
        "accounts": len(accounts),
        "tests": len(custom_tests),
        "submissions": len(submissions),
    }
