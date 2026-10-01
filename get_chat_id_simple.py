# get_chat_id_simple.py
import requests
import json

def get_chat_id_simple():
    print("🤵‍♂️ ПОЛУЧЕНИЕ CHAT_ID ДЛЯ MENTAL MANAGER")
    print("=" * 50)
    
    # Введите ваш токен
    BOT_TOKEN = input("Введите токен бота: ").strip()
    
    if not BOT_TOKEN:
        print("❌ Токен не может быть пустым!")
        return
    
    # URL для получения обновлений
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    
    try:
        print("🔍 Запрашиваю данные...")
        response = requests.get(url, timeout=10)
        data = response.json()
        
        print(f"📊 Статус: {data.get('ok', 'Unknown')}")
        
        if data['ok']:
            if data['result']:
                print("✅ Найдены сообщения!")
                print("=" * 50)
                
                # Берем последнее сообщение
                last_update = data['result'][-1]
                if 'message' in last_update:
                    chat = last_update['message']['chat']
                    chat_id = chat['id']
                    first_name = chat.get('first_name', 'Не указано')
                    username = chat.get('username', 'Не указано')
                    text = last_update['message'].get('text', 'Нет текста')
                    
                    print(f"🎯 ВАШ CHAT_ID: {chat_id}")
                    print(f"👤 Имя: {first_name}")
                    print(f"📱 Username: @{username}")
                    print(f"💬 Последнее сообщение: {text}")
                    print("=" * 50)
                    print(f"📋 СКОПИРУЙТЕ ЭТОТ НОМЕР: {chat_id}")
                    
                else:
                    print("❌ В обновлении нет сообщения")
            else:
                print("❌ Нет сообщений в истории")
                print("\n📝 ИНСТРУКЦИЯ:")
                print("1. Откройте Telegram")
                print("2. Найдите вашего бота (@your_bot_name)")
                print("3. Напишите ему ЛЮБОЕ сообщение (например, 'привет')")
                print("4. Запустите этот скрипт снова")
                print("5. Если не работает - напишите боту /start")
                
        else:
            error_desc = data.get('description', 'Неизвестная ошибка')
            print(f"❌ Ошибка Telegram API: {error_desc}")
            
            if "Not Found" in error_desc:
                print("⚠️ Проверьте правильность токена!")
            elif "Conflict" in error_desc:
                print("⚠️ Другой сервер использует этот токен!")
                
    except requests.exceptions.Timeout:
        print("❌ Таймаут запроса. Проверьте интернет-соединение")
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка соединения. Проверьте интернет")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

if __name__ == "__main__":
    get_chat_id_simple()
    
    print("\n🎯 ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:")
    print("1. Скопируйте CHAT_ID из сообщения выше")
    print("2. Откройте trainer_manager.py")
    print("3. Замените YOUR_CHAT_ID_HERE на скопированный номер")
    print("4. Запустите: python trainer_manager.py")