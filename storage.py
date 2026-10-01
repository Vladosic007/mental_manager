import json
import os
from datetime import datetime


class Storage:
    def __init__(self):
        self.profiles_file = 'user_profiles.json'
        self.states_file = 'user_states.json'
        self.moods_file = 'mood_tracking.json'

    def load_profiles(self):
        """Загружает профили пользователей"""
        if os.path.exists(self.profiles_file):
            with open(self.profiles_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_profiles(self, profiles):
        """Сохраняет профили пользователей"""
        with open(self.profiles_file, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)

    def load_states(self):
        """Загружает состояния диалогов"""
        if os.path.exists(self.states_file):
            with open(self.states_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_states(self, states):
        """Сохраняет состояния диалогов"""
        with open(self.states_file, 'w', encoding='utf-8') as f:
            json.dump(states, f, ensure_ascii=False, indent=2)

    def load_moods(self):
        """Загружает данные о настроениях"""
        if os.path.exists(self.moods_file):
            with open(self.moods_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_moods(self, moods):
        """Сохраняет данные о настроениях"""
        with open(self.moods_file, 'w', encoding='utf-8') as f:
            json.dump(moods, f, ensure_ascii=False, indent=2)

    def save_all(self, profiles, states, moods):
        """Сохраняет все данные"""
        self.save_profiles(profiles)
        self.save_states(states)
        self.save_moods(moods)
        print("💾 Все данные сохранены")