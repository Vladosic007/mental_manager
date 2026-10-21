import requests
import json

def get_updates(token):
    """Получает последние обновления для определения chat_id"""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['ok'] and data['result']:
            for update in data['result']:
                if 'message' in update:
                    chat_id = update['message']['chat']['id']
                    username = update['message']['chat'].get('username', 'N/A')
                    first_name = update['message']['chat'].get('first_name', 'N/A')
                    print(f"📱 Chat ID: {chat_id} | User: {first_name} (@{username})")
        else:
            print("❌ Нет сообщений. Сначала отправьте боту любое сообщение.")
    else:
        print(f"❌ Ошибка: {response.status_code}")

if __name__ == "__main__":
    token = input("Введите токен бота: ")
    get_updates(token)

    