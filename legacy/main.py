import os
import logging
import requests
import json
import re
import random
import asyncio
import io
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from storage import Storage

# ====== ДОБАВЛЕННЫЕ ИМПОРТЫ ДЛЯ ГРАФИКОВ ======
import matplotlib

matplotlib.use('Agg')  # Чтобы графики рисовались в памяти, без вызова графических окон Windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ====== ДОБАВЛЕННЫЕ ИМПОРТЫ ДЛЯ ML ======
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    ML_AVAILABLE = True
except ImportError:
    print("⚠️ ML-библиотеки не найдены. Установите torch, sentence-transformers, scikit-learn.")
    ML_AVAILABLE = False

# ========== ДИАГНОСТИКА В САМОМ НАЧАЛЕ ==========
print("🔍 ДИАГНОСТИКА ЗАПУСКА")
print(f"Текущая папка: {os.getcwd()}")
print(f"Файл .env существует: {os.path.exists('.env')}")

load_dotenv()

YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

print(f"YANDEX_FOLDER_ID = '{YANDEX_FOLDER_ID}'")
print(f"YANDEX_API_KEY = '{YANDEX_API_KEY[:10] if YANDEX_API_KEY else 'None'}'")
print(f"TOKEN = '{TOKEN[:10] if TOKEN else 'None'}'")
print("=" * 50)

if not TOKEN:
    print("❌ ОШИБКА: TELEGRAM_BOT_TOKEN не найден!")
    exit(1)

print(f"✅ Токен бота загружен: {TOKEN[:10]}...")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

try:
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        KNOWLEDGE_BASE = json.load(f)
    print("✅ База знаний загружена")
except Exception as e:
    print(f"⚠️ Не удалось загрузить базу знаний: {e}")
    KNOWLEDGE_BASE = {
        "learned_concepts": {},
        "techniques": [],
        "user_contributions": [],
        "books": []
    }

# СОЗДАЁМ ХРАНИЛИЩЕ ДАННЫХ
storage = Storage()
user_profiles = storage.load_profiles()
user_assessment_states = storage.load_states()
user_mood_tracking = storage.load_moods()

user_test_states = {}  # {user_id: {"test_type": str, "current_q": int, "answers": list}}

print(f"✅ Загружено {len(user_profiles)} профилей пользователей")
print(f"✅ Загружено {len(user_assessment_states)} состояний диалогов")
print(f"✅ Загружено {len(user_mood_tracking)} записей настроений")

# ====== ИНИЦИАЛИЗАЦИЯ НЕЙРОНКИ И ВЕКТОРНОЙ БАЗЫ ======
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
    "я планирую навредить себе физически"
]

if ML_AVAILABLE:
    print("⏳ Загружаем модель для семантического поиска (это может занять время при первом запуске)...")
    try:
        embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Нейронка Sentence-Transformers успешно загружена!")

        print("🧠 Подготовка векторной базы знаний...")
        for category, items in KNOWLEDGE_BASE.get("learned_concepts", {}).items():
            for item in items:
                text = f"Концепция: {item.get('concept', '')}. Описание: {item.get('description', '')}. Применение: {item.get('application', '')}"
                kb_texts.append(text)

        if kb_texts:
            kb_embeddings = embedder.encode(kb_texts)
            print(f"✅ Векторизовано {len(kb_texts)} записей из базы.")
        else:
            print("⚠️ База знаний пуста для векторизации.")

        print("🚨 Векторизуем эталоны кризисных ситуаций...")
        crisis_embeddings = embedder.encode(crisis_anchors)

    except Exception as e:
        print(f"❌ Пиздец, ошибка загрузки ML-модели или векторизации: {e}")
        embedder = None


async def get_relevant_knowledge(query, top_k=2):
    if not embedder or len(kb_texts) == 0:
        return ""

    query_emb = await asyncio.to_thread(embedder.encode, [query])
    similarities = cosine_similarity(query_emb, kb_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant = []
    for idx in top_indices:
        if similarities[idx] > 0.25:
            relevant.append(kb_texts[idx])

    if relevant:
        print(f"🔍 Найдено {len(relevant)} релевантных кусков инфы из базы знаний.")
        return "\n\nВНУТРЕННЯЯ БАЗА ЗНАНИЙ ДЛЯ АНАЛИЗА (СТРОГОЕ ПРАВИЛО: Используй эту информацию ТОЛЬКО для понимания контекста и формирования уточняющих вопросов. НЕ ВЫДАВАЙ ЭТИ ТЕХНИКИ И РЕШЕНИЯ ПОЛЬЗОВАТЕЛЮ ПРЯМО СЕЙЧАС. Твоя задача - сначала глубоко исследовать проблему):\n" + "\n".join(
            f"- {text}" for text in relevant)
    return ""


def create_user_profile(user_id):
    user_id_str = str(user_id)

    if user_id_str in user_profiles:
        return user_profiles[user_id_str]

    print(f"🆕 Создаём новый профиль для {user_id}")
    new_profile = {
        "user_id": user_id,
        "basic_info": {
            "name": "",
            "age": "",
            "gender": "",
            "occupation": "",
            "location": ""
        },
        "psychological_profile": {
            "main_issues": [],
            "triggers": [],
            "coping_style": "",
            "emotional_patterns": [],
            "strengths": [],
            "vulnerabilities": [],
            "personality_traits": [],
            "communication_style": "",
            "preferred_response_style": "analytical"
        },
        "life_context": {
            "relationships": "",
            "work_situation": "",
            "health_status": "",
            "sleep_quality": "",
            "social_support": ""
        },
        "conversation_history": [],
        "techniques_used": {},
        "progress_metrics": {
            "mood_tracking": [],
            "symptom_severity": {},
            "compliance_rate": 0,
            "improvement_areas": [],
            "breakthrough_moments": []
        },
        "assessment_data": {
            "completed_assessments": [],
            "current_problems": [],
            "goals": [],
            "previous_experience": ""
        },
        "paid_subscription": {
            "has_subscription": False,
            "subscription_type": None,
            "psychologist_access": False,
            "subscription_expires": None
        },
        "assessment_complete": False,
        "created_at": datetime.now().isoformat(),
        "last_assessment": None,
        "total_sessions": 0,
        "last_active": datetime.now().isoformat(),
        "preferences": {
            "direct_communication": True,
            "technique_focus": True,
            "homework_assignments": False
        },
        "conversation_memory": {
            "last_questions": [],
            "topics_covered": [],
            "last_topic": "",
            "user_stated_wellbeing": False
        }
    }

    user_profiles[user_id_str] = new_profile
    storage.save_profiles(user_profiles)
    return new_profile


def save_knowledge_base():
    try:
        with open('knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(KNOWLEDGE_BASE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Ошибка сохранения базы знаний: {e}")


def get_dynamic_response_prompt(user_id, user_message, topic):
    profile = user_profiles.get(str(user_id), {})
    user_name = profile.get('basic_info', {}).get('name', '')
    previous_experience = profile.get('assessment_data', {}).get('previous_experience', '')

    history_text = ""
    if profile.get('conversation_history'):
        history = profile['conversation_history'][-4:]
        for msg in history:
            role = "Студент" if msg['role'] == 'user' else "Ты"
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


def has_recently_asked(user_id, question_text, hours=24):
    user_id_str = str(user_id)
    if user_id_str not in user_profiles:
        return False

    profile = user_profiles[user_id_str]
    if "conversation_memory" not in profile:
        profile["conversation_memory"] = {
            "last_questions": [],
            "topics_covered": [],
            "last_topic": "",
            "user_stated_wellbeing": False
        }

    for q in profile["conversation_memory"]["last_questions"][-3:]:
        if question_text in q or q in question_text:
            return True

    return False


async def mood_tracking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat.id)

    mood_options = """
🎭 ОЦЕНИ СВОЕ НАСТРОЕНИЕ СЕЙЧАС:

1️⃣ - Очень плохо
2️⃣ - Плохо  
3️⃣ - Нейтрально
4️⃣ - Хорошо
5️⃣ - Отлично

Просто отправь цифру от 1 до 5, и я сохраню твое настроение для анализа.
"""
    await update.message.reply_text(mood_options)

    if user_id not in user_mood_tracking:
        user_mood_tracking[user_id] = {}
    user_mood_tracking[user_id]['waiting_for_mood'] = True


async def technique_library_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    techniques_text = """
🛠️ БИБЛИОТЕКА ТЕХНИК

Для тревоги:
🧘‍♂️ Дыхание 4-7-8 - Вдох 4, задержка 7, выдох 8
🌍 Заземление 5-4-3-2-1 - 5 вещей, 4 ощущения, 3 звука...
📝 Декстрофобия - Анализ худшего сценария

Для депрессии:
🌟 Правило 5 минут - Начать с малого
📊 Шкала активности - Связь действий и настроения
🎯 Микро-цели - Маленькие достижения

Для отношений:
💬 Я-сообщения - Конструктивное общение
👂 Активное слушание - Понимание без оценок
🕰️ Тайм-аут - Пауза в конфликтах

Для гнева:
🛑 Стоп-сигнал - Остановка реакции
💪 Физическая разрядка - Сброс напряжения
🎯 Конструктивное выражение - Письмо без отправки

Напиши название техники, чтобы узнать подробности!
"""
    await update.message.reply_text(techniques_text, parse_mode='Markdown')


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
        else:
            return "светская беседа"


async def detect_urgency_level(user_message):
    if ML_AVAILABLE and embedder is not None and len(crisis_embeddings) > 0:
        msg_emb = await asyncio.to_thread(embedder.encode, [user_message])
        similarities = cosine_similarity(msg_emb, crisis_embeddings)[0]
        max_sim = max(similarities)

        if max_sim > 0.75:
            return "CRISIS"

        return "NORMAL"

    message_lower = user_message.lower()
    crisis_phrases = ["суицид", "хочу умереть", "покончить с собой"]
    for phrase in crisis_phrases:
        if re.search(r'\b' + re.escape(phrase) + r'\b', message_lower):
            return "CRISIS"

    return "NORMAL"


def update_user_profile(user_id, user_message, bot_response, topic):
    user_id_str = str(user_id)
    if user_id_str not in user_profiles:
        user_profiles[user_id_str] = create_user_profile(user_id_str)

    profile = user_profiles[user_id_str]

    if topic and topic not in profile["psychological_profile"]["main_issues"]:
        profile["psychological_profile"]["main_issues"].append(topic)

    profile["total_sessions"] = profile.get("total_sessions", 0) + 1

    if "conversation_history" not in profile:
        profile["conversation_history"] = []


async def progress_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерация красивого графика настроения и вывод статистики"""
    user_id = str(update.message.chat.id)

    if user_id not in user_profiles:
        await update.message.reply_text("❌ У тебя еще нет профиля. Начни общение со мной!")
        return

    profile = user_profiles[user_id]
    total_sessions = profile['total_sessions']
    techniques_used = len(profile['techniques_used'])

    mood_data = profile.get('progress_metrics', {}).get('mood_tracking', [])
    mood_entries = len(mood_data)

    progress_text = f"""
📊 *ТВОЙ ПРОГРЕСС*

*Общая статистика:*
💬 Сессий с ботом: {total_sessions}
🛠️ Техник использовано: {techniques_used}
🎭 Настроений отслежено: {mood_entries}

*Темы работы:*
{', '.join(profile['psychological_profile']['main_issues']) if profile['psychological_profile']['main_issues'] else 'Еще не определены'}

*Следующие шаги:*
"""
    if total_sessions < 3:
        progress_text += "- Пройти оценку (/assessment)\n- Попробовать технику из библиотеки (/techniques)"
    elif techniques_used == 0:
        progress_text += "- Изучить библиотеку техник (/techniques)\n- Отслеживать настроение (/mood)"
    else:
        progress_text += "- Продолжать применять работающие техники\n- Анализировать прогресс регулярно"

    # ЕСЛИ ЕСТЬ МИНИМУМ 2 ЗАПИСИ НАСТРОЕНИЯ - СТРОИМ ГРАФИК
    if mood_entries >= 2:
        try:
            dates = [datetime.fromisoformat(m["timestamp"]) for m in mood_data]
            scores = [m["score"] for m in mood_data]

            plt.figure(figsize=(9, 4.5))
            plt.plot(dates, scores, marker='o', color='#4a90e2', linewidth=2, markersize=8)
            plt.ylim(0.5, 5.5)
            plt.yticks([1, 2, 3, 4, 5], ['1 (Очень плохо)', '2 (Плохо)', '3 (Нейтрально)', '4 (Хорошо)', '5 (Отлично)'])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title("Динамика твоего настроения", fontsize=14, pad=15)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
            plt.gcf().autofmt_xdate()
            plt.tight_layout()

            # Сохраняем график в буфер, чтобы не мусорить на диске
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close()

            await update.message.reply_photo(photo=buf, caption=progress_text, parse_mode='Markdown')
        except Exception as e:
            print(f"❌ Ошибка генерации графика: {e}")
            await update.message.reply_text(progress_text, parse_mode='Markdown')
    else:
        # Если записей мало, просто кидаем текст
        await update.message.reply_text(progress_text, parse_mode='Markdown')


async def quick_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🚨 *Если прямо сейчас тяжело*\n\n"
        "🌬️ *Дыхание (делай прямо сейчас):*\n"
        "Вдох на 4 счёта — задержка 4 — выдох на 8.\n"
        "Повтори 5 раз. Это реально работает.\n\n"
        "🌍 *Заземление 5-4-3-2-1:*\n"
        "— назови 5 вещей, которые видишь\n"
        "— 4 вещи, которые ощущаешь телом\n"
        "— 3 звука вокруг\n"
        "— 2 запаха\n"
        "— 1 вкус\n\n"
        "📞 *Телефоны доверия:*\n"
        "8-800-2000-122 — бесплатно, круглосуточно\n"
        "112 — экстренные службы\n\n"
        "🏫 *Психологическая служба ЮФУ:*\n"
        "Бесплатно для студентов. Без осуждения.\n"
        "Напиши мне — я помогу найти контакты.\n\n"
        "Ты справишься. Напиши мне, что происходит 💬"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def get_ai_response(user_message, user_id, context):
    user_id_str = str(user_id)
    if user_mood_tracking.get(user_id_str, {}).get('waiting_for_mood'):
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

        final_system_prompt = base_prompt + kb_context

        response = await get_yandex_gpt_response(final_system_prompt, user_message)

        if not response or response.startswith("Давай"):
            return get_smart_fallback(user_message, user_id_str)

        return response

    except Exception as e:
        print(f"Ошибка в get_ai_response: {e}")
        return get_smart_fallback(user_message, user_id_str)


def extract_name(user_message):
    message = user_message.lower()
    if "меня зовут" in message:
        parts = message.split("меня зовут")
        if len(parts) > 1:
            name_part = parts[1].strip().split()[0]
            name = ''.join(c for c in name_part if c.isalpha())
            if name:
                return name.capitalize()

    words = message.split()
    if len(words) == 2 and words[0] == "я":
        name = words[1].strip()
        if name.isalpha() and len(name) < 20:
            return name.capitalize()

    return None


def get_smart_fallback(user_message, user_id):
    user_id_str = str(user_id)
    message = user_message.lower()

    positive_phrases = ["все хорошо", "всё хорошо", "все отлично", "всё отлично",
                        "все нормально", "всё нормально", "все супер", "всё супер",
                        "ничего не беспокоит", "все в порядке", "всё в порядке"]

    if any(phrase in message for phrase in positive_phrases):
        if user_id_str in user_profiles:
            if "conversation_memory" not in user_profiles[user_id_str]:
                user_profiles[user_id_str]["conversation_memory"] = {}
            user_profiles[user_id_str]["conversation_memory"]["user_stated_wellbeing"] = True

        return random.choice([
            "Рад это слышать! Если захочешь просто поболтать - я всегда здесь.",
            "Отлично! Иногда полезно просто поделиться хорошими новостями.",
            "Здорово, что у тебя всё хорошо! Может расскажешь, что приятного произошло?",
            "Приятно слышать! Если появится что обсудить или просто поболтать - пиши."
        ])

    if any(word in message for word in ["тревож", "страх", "боюсь", "нерв", "паник"]):
        return "Расскажи подробнее, что именно вызывает эту тревогу?"

    elif any(word in message for word in
             ["груст", "плохо", "депресс", "тоск", "уныл", "нет сил", "апатия", "выгорание"]):
        return "Как долго ты уже находишься в таком состоянии? Было ли какое-то событие, с которого всё началось?"

    general_questions = [
        "Расскажи об этом поподробнее. Как именно это на тебя влияет?",
        "Давай остановимся на этом. Что конкретно тебя беспокоит в этой ситуации?",
        "Интересно. А почему, как ты думаешь, это произошло именно так?"
    ]
    return random.choice(general_questions)


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
                "description": get_mood_description(mood_score)
            }

            user_profiles[user_id_str]["progress_metrics"]["mood_tracking"].append(mood_entry)
            user_mood_tracking[user_id_str]['waiting_for_mood'] = False

            storage.save_profiles(user_profiles)
            storage.save_moods(user_mood_tracking)

            return get_mood_response(mood_score, user_id_str)
    except:
        pass

    user_mood_tracking[user_id_str]['waiting_for_mood'] = False
    return "Не смог распознать оценку. Используй цифры от 1 до 5. Хочешь попробовать снова? Отправь /mood"


def get_mood_description(score):
    descriptions = {
        1: "очень плохое",
        2: "плохое",
        3: "нейтральное",
        4: "хорошее",
        5: "отличное"
    }
    return descriptions.get(score, "неопределенное")


def get_mood_response(score, user_id):
    base_responses = {
        1: "Заметил твое состояние. Что конкретно сейчас вытянуло из тебя столько сил? Расскажи подробнее.",
        2: "Вижу, что настроение не очень. Есть ли какая-то конкретная ситуация, которая крутится в голове?",
        3: "Нейтральное состояние - это нормально. О чем бы ты хотел поговорить сегодня?",
        4: "Рад, что у тебя хорошее настроение! Что приятного сегодня случилось?",
        5: "Отлично! Что именно так подняло тебе настроение?"
    }
    return base_responses.get(score, "Спасибо за оценку! Как прошел твой день?")


async def handle_crisis_situation(user_message):
    crisis_prompt = """Студент ЮФУ написал тебе что-то тревожное. Ты — друг-помощник Ментал.
Твои действия:
1. Скажи прямо и тепло, что ты рядом
2. Предложи технику заземления (5-4-3-2-1 или дыхание)
3. Дай телефоны доверия: 8-800-2000-122 (бесплатно), 112
4. Скажи про Психологическую службу ЮФУ — там помогут
Не давай пустых обещаний. Будь конкретным и человечным."""
    return await get_yandex_gpt_response(crisis_prompt, user_message)


# ====== ПСИХОЛОГИЧЕСКИЕ ТЕСТЫ ======

STRESS_TEST = {
    "name": "Уровень стресса",
    "intro": (
        "📊 *Тест: Уровень стресса*\n\n"
        "6 вопросов — каждый займёт 5 секунд.\n"
        "Оцени каждый пункт от 1 до 5:\n"
        "*1* — почти никогда\n"
        "*2* — редко\n"
        "*3* — иногда\n"
        "*4* — часто\n"
        "*5* — почти всегда\n\n"
        "Готов? Поехали 👇"
    ),
    "questions": [
        "❓ *1/6* За последние 2 недели — как часто ты чувствовал, что времени катастрофически не хватает и ты не успеваешь?\n\n_Оцени от 1 до 5:_",
        "❓ *2/6* Как часто мысли об учёбе, дедлайнах или экзаменах мешали нормально отдохнуть или уснуть?\n\n_Оцени от 1 до 5:_",
        "❓ *3/6* Как часто ты чувствовал, что задач слишком много и ты не вывозишь?\n\n_Оцени от 1 до 5:_",
        "❓ *4/6* Как часто ты срывался на близких или чувствовал раздражение без особой причины?\n\n_Оцени от 1 до 5:_",
        "❓ *5/6* Как часто ты физически чувствовал стресс — болела голова, зажатые плечи, проблемы со сном?\n\n_Оцени от 1 до 5:_",
        "❓ *6/6* Как часто тебя накрывала мысль: «Зачем всё это, я всё равно не справляюсь»?\n\n_Оцени от 1 до 5:_"
    ]
}

BURNOUT_TEST = {
    "name": "Учебное выгорание",
    "intro": (
        "🔥 *Тест: Учебное выгорание*\n\n"
        "7 вопросов про твои ощущения от учёбы прямо сейчас.\n"
        "Отвечай честно — никто не оценивает.\n"
        "Шкала: *1* (совсем нет) до *5* (да, именно так).\n\n"
        "Погнали 👇"
    ),
    "questions": [
        "❓ *1/7* Учёба для тебя сейчас — скорее обязанность, чем что-то интересное?\n\n_От 1 (совсем нет) до 5 (да, именно так):_",
        "❓ *2/7* Ты часто откладываешь учёбу до последнего, даже зная про дедлайны?\n\n_От 1 до 5:_",
        "❓ *3/7* Ты устаёшь от учёбы, даже когда объективно не сделал ничего особо тяжёлого?\n\n_От 1 до 5:_",
        "❓ *4/7* Ты стал более цинично или безразлично относиться к парам, преподам, университету?\n\n_От 1 до 5:_",
        "❓ *5/7* Тебя часто преследует ощущение, что мог бы делать лучше, но не делаешь?\n\n_От 1 до 5:_",
        "❓ *6/7* Ты помнишь или чувствуешь, зачем вообще поступал на эту специальность?\n\n_От 1 (не помню зачем) до 5 (конечно, это важно):_",
        "❓ *7/7* Если бы мог взять паузу от учёбы прямо сейчас — взял бы?\n\n_От 1 (нет, мне и так норм) до 5 (однозначно да):_"
    ]
}

MOOD_TEST = {
    "name": "Как ты сегодня",
    "intro": (
        "😊 *Как ты сегодня?*\n\n"
        "5 быстрых вопросов — узнаем, что происходит.\n"
        "Шкала от 1 до 5.\n\n"
        "Поехали 👇"
    ),
    "questions": [
        "❓ *1/5* Оцени своё настроение прямо сейчас.\n\n_От 1 (всё плохо) до 5 (всё отлично):_",
        "❓ *2/5* Сколько ты спал прошлой ночью?\n\n1 — меньше 5 часов\n3 — 6-7 часов\n5 — 8+ часов (выспался)\n\n_Оцени от 1 до 5:_",
        "❓ *3/5* Ты нормально поел сегодня?\n\n1 — не ел совсем\n3 — перекусил на бегу\n5 — да, нормально поел\n\n_От 1 до 5:_",
        "❓ *4/5* Есть что-то конкретное, что сейчас тревожит или давит?\n\n1 — нет, всё спокойно\n3 — есть кое-что\n5 — да, и это сильно давит\n\n_От 1 до 5:_",
        "❓ *5/5* Сделал что-нибудь приятное или полезное для себя сегодня — даже самое маленькое?\n\n1 — нет, ничего\n3 — что-то есть\n5 — да, и это было здорово\n\n_От 1 до 5:_"
    ]
}

ALL_TESTS = {"stress": STRESS_TEST, "burnout": BURNOUT_TEST, "mood": MOOD_TEST}


def calculate_stress_result(answers):
    total = sum(answers)
    if total <= 12:
        level = "🟢 Стресс в норме"
        body = (
            f"*{total} баллов из 30* — всё в порядке.\n\n"
            "Продолжай в том же духе! Несколько советов, чтобы так и оставалось:\n"
            "• Не забывай про паузы — 10 минут прогулки между парами реально помогают\n"
            "• Веди список задач, чтобы дедлайны не висели в голове тяжёлым грузом\n"
            "• Хвали себя за маленькие победы — сдал зачёт, разобрался в теме, поел нормально 🙂"
        )
    elif total <= 20:
        level = "🟡 Умеренный стресс"
        body = (
            f"*{total} баллов из 30* — не критично, но стоит за собой следить.\n\n"
            "Что реально помогает:\n"
            "• *Техника «помидора»*: 25 минут работаю — 5 минут отдыхаю. Мозг устаёт от монотонности, а не от объёма\n"
            "• Запиши ВСЕ задачи на бумагу — это уже снимает часть напряжения\n"
            "• Дыхание перед сном: вдох на 4 счёта, задержка 4, выдох на 6. Работает\n"
            "• Поговори с кем-нибудь — другом, семьёй. Не обязательно про проблемы, просто пообщаться"
        )
    else:
        level = "🔴 Высокий стресс"
        body = (
            f"*{total} баллов из 30* — твой организм сигнализирует, надо остановиться.\n\n"
            "Сделай прямо сейчас:\n"
            "• *Дыхание*: вдох на 4 счёта, выдох на 8 — сделай 5 раз. Это физиология, не метафора\n"
            "• Выпиши всё, что давит — на бумаге это выглядит менее страшно\n"
            "• Определи ОДНУ самую срочную задачу. Не пять. Одну\n"
            "• Зайди в Психологическую службу ЮФУ — там помогают бесплатно, без осуждения\n"
            "• Просить помощь — это нормально, не слабость"
        )
    return f"📊 *Результат: {level}*\n\n{body}\n\nЕсли хочешь — расскажи, что сейчас давит больше всего? Разберёмся 💬"


def calculate_burnout_result(answers):
    q6_reversed = 6 - answers[5]
    adjusted = answers[:5] + [q6_reversed] + [answers[6]]
    total = sum(adjusted)

    exhaustion = answers[0] + answers[2] + answers[4]
    cynicism = answers[3] + q6_reversed
    avoidance = answers[1] + answers[6]

    dominant = max(
        ("exhaustion", exhaustion),
        ("cynicism", cynicism),
        ("avoidance", avoidance),
        key=lambda x: x[1]
    )[0]

    type_tips = {
        "exhaustion": (
            "⚡ *Главное у тебя — усталость.*\n"
            "Ты тянешь слишком много и силы на исходе. Это не лень — это физиология. "
            "Хотя бы один день без учёбы на этой неделе — это инвестиция, не потеря."
        ),
        "cynicism": (
            "💭 *Главное у тебя — потеря смысла.*\n"
            "Ты немного разочаровался в учёбе или универе. Бывает. Попробуй вспомнить, "
            "что тебя зацепило в специальности в самом начале — или поговори с кем-то старше на курсе."
        ),
        "avoidance": (
            "🚧 *Главное у тебя — избегание.*\n"
            "Ты откладываешь, потому что задачи кажутся огромными или бессмысленными. "
            "Начни с самого маленького шага — буквально 15 минут одной задачи. Потом станет легче."
        )
    }

    if total <= 14:
        return (
            "🟢 *Выгорания нет*\n\n"
            f"*{total} баллов* — в целом всё в порядке. Учёба даётся, смысл есть, энергия есть.\n\n"
            "Главное — не доводить до крайностей. Следи за режимом, делай паузы, "
            "не забывай про жизнь вне универа. Баланс — это навык, и ты его нарабатываешь 👍"
        )
    elif total <= 24:
        return (
            f"🟡 *Лёгкое выгорание* — {total} баллов\n\n"
            "Что-то идёт не так, но это ещё не глубокая яма.\n\n"
            f"{type_tips[dominant]}\n\n"
            "Практически: попробуй один «разгрузочный день» в неделю — день без чувства вины за отдых. "
            "И пообщайся с людьми вне учёбы — это перезаряжает."
        )
    else:
        return (
            f"🔴 *Серьёзное выгорание* — {total} баллов\n\n"
            "Энергия, мотивация и смысл серьёзно на нуле. Это не слабость — это сигнал, что нужно что-то менять.\n\n"
            "Что важно сделать:\n"
            "• Обратись в Психологическую службу ЮФУ — там бесплатно и без осуждения\n"
            "• Поговори с научруком или куратором про нагрузку — они люди, часто идут навстречу\n"
            "• Разреши себе замедлиться. Сдать сессию на тройку и остаться живым — лучше, чем выгореть в ноль\n\n"
            "Расскажи — что сейчас самое тяжёлое?"
        )


def calculate_mood_result(answers):
    mood, sleep, food, stress, self_care = answers
    parts = []

    if mood <= 2:
        parts.append("😔 Настроение сегодня не очень — это бывает, и это пройдёт.")
    elif mood <= 3:
        parts.append("🙂 Настроение среднее — ни шатко ни валко.")
    else:
        parts.append("😊 Настроение неплохое — уже хорошо!")

    if sleep <= 2:
        parts.append(
            "😴 Поспал мало. После 2 ночей недосыпа мозг работает как после алкоголя — это не шутка, это физиология. "
            "Постарайся поспать хотя бы 7 часов сегодня."
        )
    elif sleep >= 4:
        parts.append("💤 Со сном всё норм — это важнее, чем кажется.")

    if food <= 2:
        parts.append(
            "🍕 Еда — это топливо для мозга. Зайди в столовую или хотя бы перекуси — на голодный желудок учёба не идёт."
        )

    if stress >= 4:
        parts.append("💬 Что-то давит — хочешь расскажешь подробнее? Иногда просто выговориться уже помогает.")

    if self_care >= 4:
        parts.append("✅ Здорово, что сделал что-то для себя — это важнее, чем кажется.")
    elif self_care <= 2:
        parts.append(
            "💡 Сделай хоть что-то маленькое для себя сегодня: выйди подышать, послушай музыку, "
            "съешь что-нибудь вкусное. Это не эгоизм — это поддержание ресурса."
        )

    wellbeing = mood + sleep + food - (stress - 1) + self_care
    if wellbeing <= 10:
        summary = "\nСегодня не лучший день — и это ок. Позаботься о себе: поешь, выпей воды, сделай паузу. Если хочешь поговорить — я здесь 🤝"
    elif wellbeing <= 17:
        summary = "\nСредний день — бывает. Главное, что держишься. Если что-то давит — расскажи, разберёмся вместе."
    else:
        summary = "\nСегодня неплохой день! Держись этого состояния 💪"

    return "📊 *Как ты сегодня:*\n\n" + "\n\n".join(parts) + summary


async def start_test(update, test_type):
    user_id = str(update.message.chat.id)
    test = ALL_TESTS.get(test_type)
    if not test:
        return
    user_test_states[user_id] = {"test_type": test_type, "current_q": 0, "answers": []}
    await update.message.reply_text(test["intro"], parse_mode='Markdown')
    await asyncio.sleep(0.5)
    await update.message.reply_text(test["questions"][0], parse_mode='Markdown')


async def stress_test_command(update, context):
    await start_test(update, "stress")


async def burnout_test_command(update, context):
    await start_test(update, "burnout")


async def mood_test_command(update, context):
    await start_test(update, "mood")


async def handle_test_answer(update, user_id, user_text):
    """Обрабатывает ответ в активном тесте. Возвращает True если сообщение обработано."""
    if user_id not in user_test_states:
        return False

    state = user_test_states[user_id]
    test = ALL_TESTS[state["test_type"]]

    try:
        answer = int(user_text.strip())
        if not (1 <= answer <= 5):
            raise ValueError
    except ValueError:
        await update.message.reply_text("Введи цифру от 1 до 5 👆")
        return True

    state["answers"].append(answer)
    state["current_q"] += 1

    if state["current_q"] >= len(test["questions"]):
        calculators = {
            "stress": calculate_stress_result,
            "burnout": calculate_burnout_result,
            "mood": calculate_mood_result
        }
        result = calculators[state["test_type"]](state["answers"])
        if state["test_type"] == "mood" and user_id in user_profiles:
            mood_entry = {
                "score": state["answers"][0],
                "timestamp": datetime.now().isoformat(),
                "description": get_mood_description(state["answers"][0])
            }
            user_profiles[user_id]["progress_metrics"]["mood_tracking"].append(mood_entry)
            storage.save_profiles(user_profiles)
        del user_test_states[user_id]
        await update.message.reply_text(result, parse_mode='Markdown')
        return True

    await update.message.reply_text(test["questions"][state["current_q"]], parse_mode='Markdown')
    return True


async def get_yandex_gpt_response(prompt, user_message):
    try:
        if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
            return "Давай сосредоточимся на том, что можно сделать прямо сейчас."

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "x-folder-id": YANDEX_FOLDER_ID
        }

        req_messages = []
        if prompt:
            req_messages.append({"role": "system", "text": prompt})
        req_messages.append({"role": "user", "text": user_message[:2500]})

        data = {
            "modelUri": "gpt://b1gjgreo8r7vdsp1gapp/yandexgpt/rc",
            "completionOptions": {
                "stream": False,
                "temperature": 0.7,
                "maxTokens": 800
            },
            "messages": req_messages
        }

        def make_request():
            return requests.post(url, headers=headers, json=data, timeout=30)

        response = await asyncio.to_thread(make_request)

        if response.status_code == 200:
            result = response.json()
            return result['result']['alternatives'][0]['message']['text']
        else:
            return "Давай попробуем копнуть глубже. Что еще тебя беспокоит?"

    except Exception as e:
        return "Расскажи подробнее, как именно ты себя сейчас чувствуешь?"


async def summarize_history(history_chunk):
    """Отправляет кусок диалога в Яндекс для создания краткой выжимки (Долгосрочная память)"""
    try:
        text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history_chunk])
        prompt = """Ты - психолог-супервизор. Прочитай этот кусок диалога с клиентом и составь ОЧЕНЬ КРАТКУЮ выжимку (2-3 предложения) от третьего лица. 
Укажи главную проблему, эмоциональное состояние и важные факты. 
Не давай советов. Это пойдет в медицинскую карту для долгосрочной памяти."""

        summary = await get_yandex_gpt_response(prompt, text)
        return summary
    except Exception as e:
        print(f"❌ Ошибка саммаризации: {e}")
        return None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat.id)
    user_name = update.message.chat.first_name or "друг"

    if user_id not in user_profiles:
        user_profiles[user_id] = create_user_profile(user_id)

    if not user_profiles[user_id]["basic_info"]["name"] and user_name != "друг":
        user_profiles[user_id]["basic_info"]["name"] = user_name
        name_message = f"Я буду называть тебя {user_name}, если не возражаешь!"
    else:
        name_message = ""

    name_part = f"\n{name_message}" if name_message else ""
    start_text = (
        f"👋 *Привет, {user_name}! Я Ментал — твой помощник от ЮФУ*{name_part}\n\n"
        "Помогаю студентам справляться со стрессом сессии, дедлайнами, выгоранием "
        "и просто разобраться в том, что происходит внутри.\n\n"
        "С чего начнём?\n\n"
        "💬 *Просто напиши* — расскажи, что сейчас происходит\n\n"
        "Или выбери тест:\n"
        "📊 /stress\\_test — проверь уровень стресса\n"
        "🔥 /burnout\\_test — есть ли учебное выгорание\n"
        "😊 /mood\\_test — как ты сегодня\n\n"
        "📈 /progress — твоя динамика настроения\n"
        "🛠️ /techniques — библиотека техник\n"
        "🚨 /help — экстренная помощь"
    )
    await update.message.reply_text(start_text, parse_mode='Markdown')


async def start_assessment_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    assessment_text = "📋 *СИСТЕМА ОЦЕНКИ (В РАЗРАБОТКЕ)*"
    await update.message.reply_text(assessment_text, parse_mode='Markdown')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat.id)
    user_text = update.message.text.strip()

    if user_id not in user_profiles:
        user_profiles[user_id] = create_user_profile(user_id)

    # Если студент проходит тест — отдаём управление тест-хендлеру
    if await handle_test_answer(update, user_id, user_text):
        return

    lower_text = user_text.lower()

    if "меня зовут" in lower_text:
        parts = user_text.split("меня зовут")
        if len(parts) > 1:
            name_part = parts[1].strip().split()[0]
            name = ''.join(c for c in name_part if c.isalpha())
            if name:
                user_profiles[user_id]["basic_info"]["name"] = name
                response = f"Приятно познакомиться, {name}! Чем могу помочь?"
                user_profiles[user_id]["conversation_history"].append({
                    "role": "assistant",
                    "text": response,
                    "timestamp": datetime.now().isoformat()
                })
                storage.save_profiles(user_profiles)
                await update.message.reply_text(response)
                return

    who_questions = ["ты кто", "кто ты", "ты кто такой", "что ты такое", "ты бот", "как тебя зовут", "твое имя",
                     "как к тебе обращаться"]
    if any(q in lower_text for q in who_questions):
        response = "Я Mental Manager - твой психологический помощник. Могу помочь разобраться в мыслях и чувствах, отслеживать настроение или просто поболтать. Как к тебе обращаться?"
        user_profiles[user_id]["conversation_history"].append({
            "role": "assistant",
            "text": response,
            "timestamp": datetime.now().isoformat()
        })
        await update.message.reply_text(response)
        return

    memory_questions = ["ты помнишь меня", "помнишь меня", "ты знаешь меня", "кто я", "что ты обо мне знаешь"]
    if any(q in lower_text for q in memory_questions):
        name = user_profiles[user_id].get("basic_info", {}).get("name", "")
        if name:
            response = f"Конечно помню! Тебя зовут {name}. Как твои дела?"
        else:
            response = "Я тебя помню, но ещё не знаю, как тебя зовут! Как мне к тебе обращаться?"
        user_profiles[user_id]["conversation_history"].append(
            {"role": "assistant", "text": response, "timestamp": datetime.now().isoformat()})
        await update.message.reply_text(response)
        return

    greetings = ["привет", "здравствуй", "хай", "здаров", "приветствую", "добрый день", "добрый вечер"]
    if any(greet in lower_text for greet in greetings) and len(user_text.split()) <= 3:
        name = user_profiles[user_id].get("basic_info", {}).get("name", "")
        response = f"Привет, {name}! Как дела?" if name else "Привет! Как твои дела?"
        user_profiles[user_id]["conversation_history"].append(
            {"role": "assistant", "text": response, "timestamp": datetime.now().isoformat()})
        await update.message.reply_text(response)
        return

    await update.message.chat.send_action(action="typing")

    response = await get_ai_response(user_text, user_id, context)

    user_profiles[user_id]["conversation_history"].append({
        "role": "user",
        "text": user_text[:500],
        "timestamp": datetime.now().isoformat()
    })

    user_profiles[user_id]["conversation_history"].append({
        "role": "assistant",
        "text": response[:500],
        "timestamp": datetime.now().isoformat()
    })

    # ====== ДОЛГОСРОЧНАЯ ПАМЯТЬ: ЕСЛИ ИСТОРИЯ СТАЛА СЛИШКОМ ДЛИННОЙ ======
    if len(user_profiles[user_id]["conversation_history"]) > 16:
        print("🧠 История длинная, запускаю фоновую саммаризацию...")
        # Берем старые 10 сообщений
        older_messages = user_profiles[user_id]["conversation_history"][:10]
        recent_messages = user_profiles[user_id]["conversation_history"][10:]

        # Делаем выжимку
        summary = await summarize_history(older_messages)

        if summary and not summary.startswith("Давай"):
            old_exp = user_profiles[user_id]["assessment_data"].get("previous_experience", "")
            date_str = datetime.now().strftime('%d.%m.%Y')
            user_profiles[user_id]["assessment_data"][
                "previous_experience"] = f"{old_exp}\n- {date_str}: {summary}".strip()
            print(f"💾 Саммари успешно сохранено в профиль {user_id}")

        # Оставляем только свежие сообщения
        user_profiles[user_id]["conversation_history"] = recent_messages

    storage.save_profiles(user_profiles)
    await update.message.reply_text(response)


def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("mood", mood_tracking_command))
    application.add_handler(CommandHandler("techniques", technique_library_command))
    application.add_handler(CommandHandler("progress", progress_command))
    application.add_handler(CommandHandler("help", quick_help_command))
    application.add_handler(CommandHandler("assessment", start_assessment_conversation))
    application.add_handler(CommandHandler("stress_test", stress_test_command))
    application.add_handler(CommandHandler("burnout_test", burnout_test_command))
    application.add_handler(CommandHandler("mood_test", mood_test_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("=" * 60)
    print("🤵  МЕНТАЛ — Бот психологической поддержки ЮФУ — ЗАПУЩЕН!")
    print("📊  Тесты: /stress_test  /burnout_test  /mood_test")
    print("📈  Графики настроения: /progress")
    print("🧠  Долгосрочная память: активна")
    print("=" * 60)

    import threading
    import time

    def auto_save():
        while True:
            time.sleep(300)
            storage.save_all(user_profiles, user_assessment_states, user_mood_tracking)
            print("💾 Автосохранение выполнено")

    save_thread = threading.Thread(target=auto_save, daemon=True)
    save_thread.start()

    application.run_polling()


if __name__ == '__main__':
    main()