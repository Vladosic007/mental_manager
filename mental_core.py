"""
mental_core.py — общий «мозг» бота Ментал.

Здесь собрана вся логика, которая раньше жила внутри main.py (Telegram-бот):
промпт «Ментал», обращение к YandexGPT, детектор кризиса (ML + запасной вариант
по ключевым словам), семантический поиск по базе знаний, распознавание имени,
локальные ответы без обращения к нейросети, память диалога и долгосрочная память.

Модуль НЕ зависит от Telegram — им пользуется и сайт (webapp/server.py),
и (при желании) сам main.py. Главная точка входа — process_message().
Ключи, база знаний и файлы профилей — те же, что у Telegram-бота.
"""

import os
import sys
import re
import json
import random
import asyncio
from pathlib import Path
from datetime import datetime

# Windows-консоль (cp1251/cp866) не умеет печатать эмодзи в диагностических print —
# переводим вывод в UTF-8, иначе на старте падает UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
from dotenv import load_dotenv

from storage import Storage

# ====== ML (грузим мягко: если библиотек нет — работаем без них) ======
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ====== Конфигурация и общие данные ======
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

try:
    with open(ROOT / "knowledge_base.json", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = json.load(f)
    print("✅ [core] База знаний загружена")
except Exception as e:
    print(f"⚠️ [core] Не удалось загрузить базу знаний: {e}")
    KNOWLEDGE_BASE = {"learned_concepts": {}, "techniques": [], "user_contributions": [], "books": []}

# Общее хранилище (те же JSON-файлы, что у Telegram-бота)
storage = Storage()
user_profiles = storage.load_profiles()
user_assessment_states = storage.load_states()
user_mood_tracking = storage.load_moods()

# ====== Инициализация нейронки и векторной базы ======
embedder = None
kb_texts = []
kb_embeddings = []
crisis_embeddings = []

crisis_anchors = [
    "я хочу покончить с собой",
    "я больше не хочу жить, я устал от всего",
    "я думаю о самоубийстве",
    "мне незачем больше жить, хочу умереть",
    "у меня острая паническая атака, я задыхаюсь",
    "помогите, я сейчас умру от паники",
    "я планирую навредить себе физически",
    "мне кажется, что жить дальше нет смысла",
    "жизнь не имеет смысла, я не вижу выхода",
    "я не вижу смысла продолжать жить",
]

# Однозначные фразы-маркеры кризиса (проверяются всегда, даже без ML).
# Специально подобраны так, чтобы не срабатывать на бытовое («не хочу жить в общаге»).
CRISIS_KEYWORDS = [
    "суицид", "покончить с собой", "покончу с собой", "покончить с жизнью",
    "убить себя", "убью себя", "хочу умереть", "хочу сдохнуть",
    "не хочу больше жить", "больше не хочу жить",
    "нет смысла жить", "жить нет смысла", "смысла жить нет",
    "жить дальше нет смысла", "дальше жить нет смысла", "жить дальше смысла нет",
    "незачем жить", "жить незачем", "причинить себе вред", "навредить себе",
    "свести счёты с жизнью", "свести счеты с жизнью",
]

# Гарантированный безопасный ответ, если нейросеть недоступна (например, 403 от YandexGPT)
CRISIS_SAFE_REPLY = (
    "Слушай, я рядом. То, что ты сейчас чувствуешь — правда тяжело, и ты молодец, что написал об этом.\n\n"
    "Давай попробуем прямо сейчас немного заземлиться. Дыхание: вдохни на 4 счёта, задержи на 4, "
    "медленно выдохни на 6. Повтори 5 раз — это реально помогает телу успокоиться.\n\n"
    "И очень прошу — не оставайся с этим один. Есть люди, которые помогут прямо сейчас:\n"
    "📞 8-800-2000-122 — телефон доверия, бесплатно и круглосуточно\n"
    "📞 112 — экстренные службы\n"
    "🏫 Психологическая служба ЮФУ — бесплатно для студентов, без осуждения\n\n"
    "Ты не обязан справляться в одиночку. Расскажи, что происходит — я побуду рядом."
)

if ML_AVAILABLE:
    print("⏳ [core] Загружаем модель для семантического поиска...")
    try:
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ [core] Sentence-Transformers загружена")

        for category, items in KNOWLEDGE_BASE.get("learned_concepts", {}).items():
            for item in items:
                text = (
                    f"Концепция: {item.get('concept', '')}. "
                    f"Описание: {item.get('description', '')}. "
                    f"Применение: {item.get('application', '')}"
                )
                kb_texts.append(text)

        if kb_texts:
            kb_embeddings = embedder.encode(kb_texts)
            print(f"✅ [core] Векторизовано {len(kb_texts)} записей базы знаний")

        crisis_embeddings = embedder.encode(crisis_anchors)
        print("✅ [core] Эталоны кризиса векторизованы")
    except Exception as e:
        print(f"❌ [core] Ошибка загрузки ML: {e}")
        embedder = None
else:
    print("ℹ️ [core] ML-библиотеки не найдены — кризис по ключевым словам, поиск по базе знаний отключён")


# ====== Профиль пользователя ======
def create_user_profile(user_id):
    user_id_str = str(user_id)
    if user_id_str in user_profiles:
        return user_profiles[user_id_str]

    new_profile = {
        "user_id": user_id,
        "basic_info": {"name": "", "age": "", "gender": "", "occupation": "", "location": ""},
        "psychological_profile": {
            "main_issues": [], "triggers": [], "coping_style": "", "emotional_patterns": [],
            "strengths": [], "vulnerabilities": [], "personality_traits": [],
            "communication_style": "", "preferred_response_style": "analytical",
        },
        "life_context": {
            "relationships": "", "work_situation": "", "health_status": "",
            "sleep_quality": "", "social_support": "",
        },
        "conversation_history": [],
        "techniques_used": {},
        "progress_metrics": {
            "mood_tracking": [], "symptom_severity": {}, "compliance_rate": 0,
            "improvement_areas": [], "breakthrough_moments": [],
        },
        "assessment_data": {
            "completed_assessments": [], "current_problems": [], "goals": [], "previous_experience": "",
        },
        "paid_subscription": {
            "has_subscription": False, "subscription_type": None,
            "psychologist_access": False, "subscription_expires": None,
        },
        "assessment_complete": False,
        "created_at": datetime.now().isoformat(),
        "last_assessment": None,
        "total_sessions": 0,
        "last_active": datetime.now().isoformat(),
        "preferences": {"direct_communication": True, "technique_focus": True, "homework_assignments": False},
        "conversation_memory": {"last_questions": [], "topics_covered": [], "last_topic": "", "user_stated_wellbeing": False},
    }

    user_profiles[user_id_str] = new_profile
    storage.save_profiles(user_profiles)
    return new_profile


# ====== Промпт «Ментал» (динамический) ======
def get_dynamic_response_prompt(user_id, user_message, topic):
    profile = user_profiles.get(str(user_id), {})
    user_name = profile.get("basic_info", {}).get("name", "")
    previous_experience = profile.get("assessment_data", {}).get("previous_experience", "")

    history_text = ""
    if profile.get("conversation_history"):
        history = profile["conversation_history"][-4:]
        for msg in history:
            role = "Студент" if msg["role"] == "user" else "Ты"
            history_text += f"{role}: {msg['text']}\n"

    base_prompt = f"""Ты — Ментал, психологический помощник для студентов Южного федерального университета (ЮФУ) в Ростове-на-Дону.

КТО ТЫ:
Ты как старший товарищ, который сам прошёл через студенчество. Знаешь, что такое январская сессия, дедлайны по курсовым, строгие преподы, жизнь в общаге, очереди в студенческой столовой. Говоришь живо и по-человечески — без казёнщины и психологических терминов.

ПРАВИЛА (строго соблюдай):

1. НЕ СТАВЬ ДИАГНОЗЫ. Никогда не говори «у тебя тревожное расстройство», «похоже на депрессию», «это симптомы...». Ты не врач — ты друг. Вместо диагнозов — поддержка и конкретная помощь.

2. СЛУШАЙ СНАЧАЛА. Если студент жалуется — задай 1-2 уточняющих вопроса прежде чем предлагать советы. Не спеши с техниками — сначала пойми суть проблемы.

3. РЕЖИМ БОЛТОВНИ. Если студент просто здоровается или болтает — будь лёгким собеседником. Не ищи скрытых проблем там, где их нет.

4. ПРАКТИЧНОСТЬ. После выслушивания предлагай конкретное: технику дыхания, способ распланировать дедлайны, как поговорить с научруком, где передохнуть между парами.

5. ВСЕГДА ЗАКАНЧИВАЙ КОНСТРУКТИВНО. Не обрывай на проблеме — дай ощущение, что выход есть и ты рядом.

6. ЗАПРЕЩЁННЫЕ ФРАЗЫ: «я понимаю тебя», «мне жаль», «это должно быть тяжело», «ты не один». Это звучит как скрипт. Говори живее.

7. КОНТЕКСТ ЮФУ (используй когда уместно):
   - Сессия в январе и июне — главный стресс-триггер для студентов
   - «Автомат» — освобождение от экзамена за активность, мечта студента
   - Научрук — научный руководитель, может быть как поддержкой, так и источником стресса
   - Общага — жизнь там требует энергии и своей психологической устойчивости
   - Курсовая / диплом — длинные дедлайны, которые пугают и прокрастинируются
   - Стипендия — мотивация учиться на хорошие оценки, особенно повышенная
   - Психологическая служба ЮФУ — реально существует, бесплатная, направляй туда при серьёзных проблемах

8. СТИЛЬ. Живой, тёплый, иногда с мягким юмором. Можно «слушай», «ну и», «давай разберёмся», «вот что попробуй». Как переписка с другом, а не официальное письмо.

ТЕКУЩАЯ ТЕМА: {topic}
ИМЯ СТУДЕНТА: {user_name if user_name else 'не представился'}

{'ПРОШЛЫЕ РАЗГОВОРЫ (кратко, чтобы не повторять одни и те же вопросы):\n' + previous_experience + '\n\n' if previous_experience else ''}

ИСТОРИЯ ДИАЛОГА:
{history_text if history_text else 'Начало разговора.'}"""

    return base_prompt


# ====== Определение темы ======
def detect_topic_with_nuance(user_message):
    message_lower = user_message.lower()

    emotional_words = ["чувствую", "ощущаю", "переживаю", "эмоции", "чувства"]
    problem_words = ["проблема", "трудность", "сложность", "не получается"]
    relationship_words = ["отношения", "друг", "партнер", "семья", "коллеги"]
    work_words = ["работа", "карьера", "начальник", "коллеги", "проект", "заказчик", "клиент"]
    anxiety_words = ["тревога", "страх", "волнение", "паника", "нервы"]
    depression_words = ["депрессия", "апатия", "нет сил", "бессмысленно", "устал", "выгорание"]

    if any(word in message_lower for word in anxiety_words):
        return "тревога и страхи"
    elif any(word in message_lower for word in depression_words):
        return "настроение и мотивация"
    elif any(word in message_lower for word in relationship_words):
        return "отношения"
    elif any(word in message_lower for word in work_words):
        return "работа и карьера"
    elif any(word in message_lower for word in emotional_words):
        return "эмоции и чувства"
    elif any(word in message_lower for word in problem_words):
        return "проблемы и решения"
    else:
        if len(user_message.split()) > 10:
            return "глубокий разбор ситуации"
        return "светская беседа"


# ====== Детектор кризиса (ML + запасной вариант) ======
async def detect_urgency_level(user_message):
    # 1. Явные фразы-маркеры — проверяем всегда (работает даже без ML и без интернета)
    message_lower = user_message.lower()
    for phrase in CRISIS_KEYWORDS:
        if phrase in message_lower:
            return "CRISIS"

    # 2. Семантическая близость к эталонам кризиса (если ML доступен)
    if ML_AVAILABLE and embedder is not None and len(crisis_embeddings) > 0:
        msg_emb = await asyncio.to_thread(embedder.encode, [user_message])
        similarities = cosine_similarity(msg_emb, crisis_embeddings)[0]
        if max(similarities) > 0.68:
            return "CRISIS"

    return "NORMAL"


# ====== Семантический поиск по базе знаний ======
async def get_relevant_knowledge(query, top_k=2):
    if not embedder or len(kb_texts) == 0:
        return ""

    query_emb = await asyncio.to_thread(embedder.encode, [query])
    similarities = cosine_similarity(query_emb, kb_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant = [kb_texts[idx] for idx in top_indices if similarities[idx] > 0.25]
    if relevant:
        return (
            "\n\nВНУТРЕННЯЯ БАЗА ЗНАНИЙ ДЛЯ АНАЛИЗА (СТРОГОЕ ПРАВИЛО: Используй эту информацию "
            "ТОЛЬКО для понимания контекста и формирования уточняющих вопросов. НЕ ВЫДАВАЙ ЭТИ "
            "ТЕХНИКИ И РЕШЕНИЯ ПОЛЬЗОВАТЕЛЮ ПРЯМО СЕЙЧАС. Твоя задача - сначала глубоко исследовать "
            "проблему):\n" + "\n".join(f"- {text}" for text in relevant)
        )
    return ""


# ====== Распознавание имени ======
def extract_name(user_message):
    message = user_message.lower()
    if "меня зовут" in message:
        parts = message.split("меня зовут")
        if len(parts) > 1:
            tail = parts[1].strip().split()
            if tail:
                name = "".join(c for c in tail[0] if c.isalpha())
                if name:
                    return name.capitalize()

    words = message.split()
    if len(words) == 2 and words[0] == "я":
        name = words[1].strip()
        if name.isalpha() and len(name) < 20:
            return name.capitalize()
    return None


# ====== Запасные ответы без нейросети ======
# Тематические ответы «Ментала» на случай, если нейросеть недоступна.
# Подобраны под студента ЮФУ: тёплые, с уточняющим вопросом и практикой.
FALLBACK_TOPICS = [
    (["экзамен", "сессия", "зачёт", "зачет", "билет", "препод", "сдать", "пересдач"], [
        "Экзамены — это всегда нервно. Расскажи, что сейчас пугает больше: что не успеешь выучить, или сам момент перед преподом?",
        "Слушай, сессия давит на всех. Давай разложим: сколько у тебя времени до экзамена и что именно нужно повторить? Иногда просто план в голове уже снимает половину паники.",
        "Понимаю, перед экзаменом мандраж — это нормально. Попробуй дыхание 4-7-8, оно реально снижает тревогу. А что сейчас самое сложное — объём материала или страх провала?",
    ]),
    (["прокрастин", "откладыва", "не могу начать", "не могу заставить", "залипа", "лень", "тяну"], [
        "Прокрастинация — это почти всегда не лень, а страх или усталость. Что именно ты откладываешь? Давай разберёмся, что мешает начать.",
        "Знакомая штука. Попробуй правило 2 минут: скажи себе «делаю всего 2 минуты» — обычно дальше втягиваешься. А что за задача, которую никак не начать?",
        "Залипнуть в телефоне вместо учёбы — это мозг ищет, где полегче. Давай попробуем «помидор»: 25 минут только задача, телефон в другой комнате. С чего начнём?",
    ]),
    (["дедлайн", "горит", "не успева", "срок", "завтра сдавать", "последний день"], [
        "Окей, дедлайн горит — давай без паники и по делу. Что конкретно нужно сдать и к какому времени?",
        "Когда горит, важно выбрать ОДНО самое важное действие прямо сейчас. Что нужно сделать в первую очередь, чтобы хоть что-то сдать?",
    ]),
    (["научрук", "научный руковод", "курсов", "диплом", "вкр"], [
        "Научрук и курсовая — это отдельный вид стресса. Что сейчас напрягает: он слишком требовательный, не отвечает, или ты сам не знаешь, с чего начать?",
        "С научруком бывает тяжело, но помни — он тоже человек. Часто помогает написать ему конкретные вопросы списком. А что именно сейчас буксует с работой?",
    ]),
    (["общаг", "общежит", "сосед", "комнат"], [
        "Жизнь в общаге — испытание не для всех. Что происходит: конфликт с соседями, шум, не получается отдохнуть?",
        "В общаге сложно, когда нет своего угла и тишины. Расскажи подробнее, что именно выматывает?",
    ]),
    (["не сплю", "не высыпа", "бессонниц", "мало сплю", "не могу уснуть"], [
        "Со сном беда — это серьёзно, недосып бьёт по всему. Из-за чего не спится: мысли крутятся или режим сбился?",
        "Когда не спишь, всё кажется тяжелее, чем есть. Что мешает уснуть — тревога, телефон, или просто не можешь выключить голову?",
    ]),
    (["одинок", "никто не понима", "нет друзей", "один", "не с кем"], [
        "Чувство одиночества в универе — чаще, чем кажется, просто об этом молчат. Расскажи, давно так себя чувствуешь?",
        "Это тяжело, когда вокруг люди, а всё равно одиноко. Что для тебя сейчас самое трудное в этом?",
    ]),
    (["тревож", "страх", "боюсь", "нерв", "паник", "волну", "переживаю"], [
        "Тревога — это нормальная реакция, хоть и неприятная. Расскажи, что именно её вызывает? Так проще понять, с чем работать.",
        "Когда тревожно, помогает вернуться в тело: вдох на 4 счёта, задержка 4, выдох на 6. Попробуй пару раз. А что сейчас больше всего накручивает?",
    ]),
    (["груст", "плохо", "депресс", "тоск", "уныл", "нет сил", "апати", "выгор", "устал", "вымота"], [
        "Слышу тебя. Когда нет сил — это сигнал, что ты давно тянешь на себе много. Давно так себя чувствуешь?",
        "Усталость и апатия накапливаются незаметно. Было какое-то событие, с которого стало тяжелее, или навалилось постепенно?",
    ]),
    (["отношен", "расста", "парень", "девушк", "любовь", "ссор"], [
        "Отношения сильно влияют на состояние. Расскажи, что произошло? Я рядом, давай разберёмся.",
        "Это важно — то, что ты переживаешь. Что сейчас тяжелее всего в этой ситуации?",
    ]),
]


def get_smart_fallback(user_message, user_id):
    user_id_str = str(user_id)
    message = user_message.lower()

    positive_phrases = [
        "все хорошо", "всё хорошо", "все отлично", "всё отлично", "все нормально",
        "всё нормально", "все супер", "всё супер", "ничего не беспокоит",
        "все в порядке", "всё в порядке", "просто зашёл", "просто зашел", "просто смотрю",
    ]
    if any(phrase in message for phrase in positive_phrases):
        if user_id_str in user_profiles:
            user_profiles[user_id_str].setdefault("conversation_memory", {})["user_stated_wellbeing"] = True
        return random.choice([
            "Рад это слышать! Если захочешь просто поболтать — я всегда здесь.",
            "Отлично! Иногда полезно просто поделиться хорошими новостями. Как настроение в целом?",
            "Здорово, что всё хорошо! Можешь просто осмотреться, а если что — пиши, обсудим что угодно.",
        ])

    for keywords, replies in FALLBACK_TOPICS:
        if any(word in message for word in keywords):
            return random.choice(replies)

    return random.choice([
        "Расскажи об этом поподробнее. Как это на тебя влияет?",
        "Давай остановимся на этом. Что конкретно тебя беспокоит в этой ситуации?",
        "Понимаю, это важно. А что сейчас для тебя самое трудное в этом?",
    ])


# ====== Настроение ======
def get_mood_description(score):
    return {1: "очень плохое", 2: "плохое", 3: "нейтральное", 4: "хорошее", 5: "отличное"}.get(score, "неопределенное")


def get_mood_response(score, user_id):
    base_responses = {
        1: "Заметил твое состояние. Что конкретно сейчас вытянуло из тебя столько сил? Расскажи подробнее.",
        2: "Вижу, что настроение не очень. Есть ли какая-то конкретная ситуация, которая крутится в голове?",
        3: "Нейтральное состояние - это нормально. О чем бы ты хотел поговорить сегодня?",
        4: "Рад, что у тебя хорошее настроение! Что приятного сегодня случилось?",
        5: "Отлично! Что именно так подняло тебе настроение?",
    }
    return base_responses.get(score, "Спасибо за оценку! Как прошел твой день?")


def handle_mood_input(user_message, user_id):
    user_id_str = str(user_id)
    try:
        mood_score = int(user_message.strip())
        if 1 <= mood_score <= 5:
            if user_id_str not in user_profiles:
                user_profiles[user_id_str] = create_user_profile(user_id_str)
            mood_entry = {
                "score": mood_score,
                "timestamp": datetime.now().isoformat(),
                "description": get_mood_description(mood_score),
            }
            user_profiles[user_id_str]["progress_metrics"]["mood_tracking"].append(mood_entry)
            user_mood_tracking.setdefault(user_id_str, {})["waiting_for_mood"] = False
            storage.save_profiles(user_profiles)
            storage.save_moods(user_mood_tracking)
            return get_mood_response(mood_score, user_id_str)
    except Exception:
        pass
    user_mood_tracking.setdefault(user_id_str, {})["waiting_for_mood"] = False
    return "Не смог распознать оценку. Используй цифры от 1 до 5."


# ====== Кризисная ситуация ======
async def handle_crisis_situation(user_message):
    crisis_prompt = """Студент ЮФУ написал тебе что-то тревожное. Ты — друг-помощник Ментал.
Твои действия:
1. Скажи прямо и тепло, что ты рядом
2. Предложи технику заземления (5-4-3-2-1 или дыхание)
3. Дай телефоны доверия: 8-800-2000-122 (бесплатно), 112
4. Скажи про Психологическую службу ЮФУ — там помогут
Не давай пустых обещаний. Будь конкретным и человечным."""
    reply = await get_yandex_gpt_response(crisis_prompt, user_message)
    # Если нейросеть недоступна (403/ошибка) — отдаём гарантированный безопасный ответ,
    # безопасность не должна зависеть от внешнего API.
    if not reply or reply.startswith(("Давай", "Расскажи")):
        return CRISIS_SAFE_REPLY
    # Гарантия: телефон доверия должен быть в ответе всегда
    if "8-800-2000-122" not in reply:
        reply += "\n\n📞 Телефон доверия: 8-800-2000-122 (бесплатно, круглосуточно), 112 — экстренные службы."
    return reply


# ====== Запрос к YandexGPT ======
async def get_yandex_gpt_response(prompt, user_message):
    try:
        if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
            return "Давай сосредоточимся на том, что можно сделать прямо сейчас."

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "x-folder-id": YANDEX_FOLDER_ID,
        }
        req_messages = []
        if prompt:
            req_messages.append({"role": "system", "text": prompt})
        req_messages.append({"role": "user", "text": user_message[:2500]})

        data = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/rc",
            "completionOptions": {"stream": False, "temperature": 0.7, "maxTokens": 800},
            "messages": req_messages,
        }

        def make_request():
            return requests.post(url, headers=headers, json=data, timeout=30)

        response = await asyncio.to_thread(make_request)
        if response.status_code == 200:
            return response.json()["result"]["alternatives"][0]["message"]["text"]
        # Показываем реальную причину в консоли (неверный ключ, кончились токены и т.п.)
        print(f"[core] YandexGPT HTTP {response.status_code}: {response.text[:300]}")
        return "Давай попробуем копнуть глубже. Что еще тебя беспокоит?"
    except Exception as e:
        print(f"[core] YandexGPT ошибка запроса: {e}")
        return "Расскажи подробнее, как именно ты себя сейчас чувствуешь?"


# ====== Долгосрочная память (саммаризация) ======
async def summarize_history(history_chunk):
    try:
        text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history_chunk])
        prompt = (
            "Ты - психолог-супервизор. Прочитай этот кусок диалога с клиентом и составь ОЧЕНЬ КРАТКУЮ "
            "выжимку (2-3 предложения) от третьего лица. Укажи главную проблему, эмоциональное состояние "
            "и важные факты. Не давай советов. Это пойдет в медицинскую карту для долгосрочной памяти."
        )
        return await get_yandex_gpt_response(prompt, text)
    except Exception:
        return None


# ====== Генерация ответа ИИ (аналог get_ai_response) ======
async def get_ai_response(user_message, user_id):
    user_id_str = str(user_id)
    if user_mood_tracking.get(user_id_str, {}).get("waiting_for_mood"):
        return handle_mood_input(user_message, user_id_str)

    urgency = await detect_urgency_level(user_message)
    if urgency == "CRISIS":
        return await handle_crisis_situation(user_message)

    if "меня зовут" in user_message.lower():
        name = extract_name(user_message)
        if name and user_id_str in user_profiles:
            user_profiles[user_id_str]["basic_info"]["name"] = name
            return f"Приятно познакомиться, {name}! О чём хочешь поговорить?"

    try:
        topic = detect_topic_with_nuance(user_message)
        base_prompt = get_dynamic_response_prompt(user_id_str, user_message, topic)
        kb_context = await get_relevant_knowledge(user_message)
        response = await get_yandex_gpt_response(base_prompt + kb_context, user_message)
        if not response or response.startswith("Давай"):
            return get_smart_fallback(user_message, user_id_str)
        return response
    except Exception as e:
        print(f"[core] Ошибка в get_ai_response: {e}")
        return get_smart_fallback(user_message, user_id_str)


def _append_history(user_id, role, text):
    user_profiles[user_id]["conversation_history"].append(
        {"role": role, "text": text, "timestamp": datetime.now().isoformat()}
    )


# ====== ГЛАВНАЯ ТОЧКА ВХОДА (аналог handle_message, но возвращает текст) ======
async def process_message(user_id, user_text):
    """Полный цикл обработки сообщения — как в Telegram-боте, но возвращает текст ответа."""
    user_id = str(user_id)
    user_text = (user_text or "").strip()
    if not user_text:
        return "Напиши что-нибудь — я слушаю."

    if user_id not in user_profiles:
        user_profiles[user_id] = create_user_profile(user_id)

    lower_text = user_text.lower()

    # 1. «Меня зовут ...»
    if "меня зовут" in lower_text:
        parts = user_text.split("меня зовут")
        if len(parts) > 1:
            tail = parts[1].strip().split()
            if tail:
                name = "".join(c for c in tail[0] if c.isalpha())
                if name:
                    user_profiles[user_id]["basic_info"]["name"] = name
                    response = f"Приятно познакомиться, {name}! Чем могу помочь?"
                    _append_history(user_id, "assistant", response)
                    storage.save_profiles(user_profiles)
                    return response

    # 2. «Кто ты?»
    who_questions = ["ты кто", "кто ты", "ты кто такой", "что ты такое", "ты бот",
                     "как тебя зовут", "твое имя", "как к тебе обращаться"]
    if any(q in lower_text for q in who_questions):
        response = ("Я Ментал — твой психологический помощник от ЮФУ. Могу помочь разобраться в мыслях "
                    "и чувствах, отследить настроение или просто поболтать. Как к тебе обращаться?")
        _append_history(user_id, "assistant", response)
        storage.save_profiles(user_profiles)
        return response

    # 3. «Помнишь меня?»
    memory_questions = ["ты помнишь меня", "помнишь меня", "ты знаешь меня", "кто я",
                        "что ты обо мне знаешь", "как меня зовут", "помнишь как меня",
                        "моё имя", "мое имя", "помнишь моё имя", "помнишь мое имя"]
    if any(q in lower_text for q in memory_questions):
        name = user_profiles[user_id].get("basic_info", {}).get("name", "")
        if name:
            response = f"Конечно помню! Тебя зовут {name}. Как твои дела?"
        else:
            response = "Я тебя помню, но ещё не знаю, как тебя зовут! Как мне к тебе обращаться?"
        _append_history(user_id, "assistant", response)
        storage.save_profiles(user_profiles)
        return response

    # 4. Приветствие (короткое)
    greetings = ["привет", "здравствуй", "хай", "здаров", "приветствую", "добрый день", "добрый вечер"]
    if any(greet in lower_text for greet in greetings) and len(user_text.split()) <= 3:
        name = user_profiles[user_id].get("basic_info", {}).get("name", "")
        response = f"Привет, {name}! Как дела?" if name else "Привет! Как твои дела?"
        _append_history(user_id, "assistant", response)
        storage.save_profiles(user_profiles)
        return response

    # 5. Полноценный ответ через нейросеть (+ кризис, тема, база знаний)
    response = await get_ai_response(user_text, user_id)

    _append_history(user_id, "user", user_text[:500])
    _append_history(user_id, "assistant", response[:500])

    # 6. Долгосрочная память: если история слишком длинная — сжимаем старое
    if len(user_profiles[user_id]["conversation_history"]) > 16:
        older_messages = user_profiles[user_id]["conversation_history"][:10]
        recent_messages = user_profiles[user_id]["conversation_history"][10:]
        summary = await summarize_history(older_messages)
        if summary and not summary.startswith("Давай"):
            old_exp = user_profiles[user_id]["assessment_data"].get("previous_experience", "")
            date_str = datetime.now().strftime("%d.%m.%Y")
            user_profiles[user_id]["assessment_data"]["previous_experience"] = f"{old_exp}\n- {date_str}: {summary}".strip()
        user_profiles[user_id]["conversation_history"] = recent_messages

    storage.save_profiles(user_profiles)
    return response
