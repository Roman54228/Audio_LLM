#!/usr/bin/env python3
"""
Скрипт для тестирования simple_api
"""

import requests
import json
import os
from pathlib import Path

# Настройки
API_URL = "http://localhost:8000"
AUDIO_FILE = "tone/ml_audi.wav"

def test_health():
    """Проверка состояния API"""
    print("🔍 Проверяем health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API работает!")
            print(f"   Статус: {data['status']}")
            print(f"   Pipeline загружен: {data['pipeline_loaded']}")
            print(f"   Время работы: {data['uptime']:.2f}с")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Не удается подключиться к API. Убедитесь, что сервер запущен на порту 8000")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_transcribe(audio_file):
    """Тестирование транскрипции"""
    print(f"\n🎤 Тестируем транскрипцию файла: {audio_file}")
    
    # Проверяем существование файла
    if not os.path.exists(audio_file):
        print(f"❌ Файл {audio_file} не найден!")
        return False
    
    try:
        # Отправляем файл
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            data = {'language': 'ru'}
            
            print("📤 Отправляем запрос...")
            response = requests.post(
                f"{API_URL}/transcribe",
                files=files,
                data=data,
                timeout=60  # 60 секунд на обработку
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Транскрипция успешна!")
            print("\n📝 Результат:")
            
            if result:
                for i, phrase in enumerate(result, 1):
                    print(f"   {i}. {phrase['text']}")
                    print(f"      Время: {phrase['start_time']:.2f}с - {phrase['end_time']:.2f}с")
            else:
                print("   (Текст не распознан)")
            
            return True
        else:
            print(f"❌ Ошибка транскрипции: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Детали: {error_data}")
            except:
                print(f"   Ответ: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Таймаут запроса. Файл слишком большой или сервер перегружен")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    """Основная функция"""
    print("🚀 Тестирование Simple API")
    print("=" * 50)
    
    # Проверяем health
    if not test_health():
        print("\n❌ API недоступен. Запустите сервер:")
        print("   docker run -p 8000:8000 -v ./models:/models t-one")
        return
    
    # Тестируем транскрипцию
    if test_transcribe(AUDIO_FILE):
        print("\n🎉 Все тесты прошли успешно!")
    else:
        print("\n❌ Тест транскрипции не прошел")

if __name__ == "__main__":
    main()
