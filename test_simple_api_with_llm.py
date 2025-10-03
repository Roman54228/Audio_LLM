#!/usr/bin/env python3
"""
Скрипт для тестирования simple_api с отправкой результата в YandexGPT
"""

import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройки
API_URL = "http://localhost:8000"
AUDIO_FILE = "tone/dropout.wav"

# Настройки YandexGPT
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "b1gd0b1ls413o390fmqk")

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
        return None
    
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
                timeout=300  # 60 секунд на обработку
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
            
            return result
        else:
            print(f"❌ Ошибка транскрипции: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Детали: {error_data}")
            except:
                print(f"   Ответ: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Таймаут запроса. Файл слишком большой или сервер перегружен")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def correct_transcription_errors(transcribed_text):
    """Исправляет опечатки в транскрибированном тексте через YandexGPT"""
    print(f"\n🔧 Исправляем опечатки в транскрипции...")
    
    # Объединяем весь текст
    full_text = " ".join([phrase['text'] for phrase in transcribed_text])
    
    try:
        # Проверяем наличие API ключа
        if not YANDEX_API_KEY:
            print("❌ YANDEX_API_KEY не найден в переменных окружения!")
            return full_text
        
        # Формируем запрос для исправления опечаток
        payload = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.1,  # Низкая температура для точности
                "maxTokens": "1000"
            },
            "messages": [
                {
                    "role": "system",
                    "text": "Ты помощник для исправления опечаток в транскрибированном тексте. Твоя задача - исправить только явные опечатки и ошибки распознавания речи, сохранив оригинальный смысл и стиль речи. НЕ добавляй новые слова, НЕ меняй структуру предложений. Просто исправь очевидные ошибки."
                },
                {
                    "role": "user",
                    "text": f"Исправь опечатки в этом транскрибированном тексте:\n\n{full_text}\n\nВерни только исправленный текст без дополнительных комментариев."
                }
            ]
        }
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {YANDEX_API_KEY}"
        }
        
        print(f"📤 Отправляем запрос на исправление опечаток...")
        
        # Отправляем запрос
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result and 'alternatives' in result['result']:
                corrected_text = result['result']['alternatives'][0]['message']['text'].strip()
                
                print("✅ Опечатки исправлены!")
                print(f"📝 Оригинал: {full_text}")
                print(f"✏️  Исправлено: {corrected_text}")
                
                return corrected_text
            else:
                print("❌ Ошибка: неожиданный формат ответа от YandexGPT.")
                return full_text
        else:
            print(f"❌ Ошибка API YandexGPT: {response.status_code} - {response.text}")
            return full_text
        
    except requests.exceptions.Timeout:
        print("⏰ Превышено время ожидания ответа от YandexGPT (15 секунд)")
        return full_text
    except Exception as e:
        print(f"❌ Ошибка при исправлении опечаток: {e}")
        return full_text

def send_to_yandex_gpt(transcribed_text, corrected_text):
    """Отправляет исправленный транскрибированный текст в YandexGPT для анализа"""
    print(f"\n🤖 Отправляем исправленный текст в YandexGPT для анализа...")
    
    try:
        # Проверяем наличие API ключа
        if not YANDEX_API_KEY:
            print("❌ YANDEX_API_KEY не найден в переменных окружения!")
            return None
        
        # Формируем запрос к YandexGPT (синтаксис из yandex_gpt_helper.py)
        payload = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "2000"
            },
            "messages": [
                {
                    "role": "system",
                    "text": "Ты опытный Senior ML engineer, который проводит дружелюбные технические собеседования. Твоя задача - дать конструктивную обратную связь по ответу кандидата.\n\nТы должен:\n1. Начать с положительных моментов в ответе\n2. Сосредоточиться на СОДЕРЖАНИИ ответа, а НЕ на структуре изложения\n3. Указать на важные концепции, которые кандидат НЕ упомянул\n4. Дать практические советы по углублению знаний\n5. Оценить общий уровень понимания темы\n6. В КОНЦЕ кратко описать, как должен выглядеть идеальный ответ на этот вопрос\n\nНЕ критикуй структуру изложения, стиль речи или мелкие неточности. Фокусируйся на том, что кандидат знает/не знает по теме, какие важные аспекты пропустил, какие интересные детали мог бы добавить."
                },
                {
                    "role": "user",
                    "text": f"Вот ответ кандидата на технический вопрос по ML (транскрибированный с аудио и исправленный от опечаток):\n\n{corrected_text}\n\nДай конструктивную обратную связь по этому ответу. Начни с положительных моментов, затем укажи на важные концепции, которые кандидат не упомянул, и дай советы по углублению знаний. НЕ критикуй структуру изложения. В конце кратко опиши, как должен выглядеть идеальный ответ на этот вопрос."
                }
            ]
        }
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {YANDEX_API_KEY}"
        }
        
        print(f"📤 Отправляем в YandexGPT...")
        
        # Отправляем запрос
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result and 'alternatives' in result['result']:
                llm_response = result['result']['alternatives'][0]['message']['text']
                
                print("✅ YandexGPT ответил!")
                print(f"\n🧠 Ответ YandexGPT:")
                print(f"   {llm_response}")
                
                return llm_response
            else:
                print("❌ Ошибка: неожиданный формат ответа от YandexGPT.")
                return None
        else:
            print(f"❌ Ошибка API YandexGPT: {response.status_code} - {response.text}")
            return None
        
    except requests.exceptions.Timeout:
        print("⏰ Превышено время ожидания ответа от YandexGPT (20 секунд)")
        return None
    except Exception as e:
        print(f"❌ Ошибка при обращении к YandexGPT: {e}")
        return None

def main():
    """Основная функция"""
    print("🚀 Тестирование Simple API с YandexGPT")
    print("=" * 50)
    
    # Проверяем настройки YandexGPT
    if not YANDEX_API_KEY:
        print("❌ Не настроен API ключ для YandexGPT!")
        print("   Установите переменную окружения YANDEX_API_KEY")
        print("   Например: export YANDEX_API_KEY='your-api-key'")
        return
    
    # Проверяем health
    if not test_health():
        print("\n❌ API недоступен. Запустите сервер:")
        print("   docker run -p 8000:8000 -v ./models:/models t-one")
        return
    
    # Тестируем транскрипцию
    transcribed_result = test_transcribe(AUDIO_FILE)
    
    if transcribed_result:
        # Сначала исправляем опечатки
        corrected_text = correct_transcription_errors(transcribed_result)
        
        # Затем отправляем исправленный текст в YandexGPT для анализа
        llm_response = send_to_yandex_gpt(transcribed_result, corrected_text)
        
        if llm_response:
            print("\n🎉 Полный цикл выполнен успешно!")
            print("   Аудио → Текст → Исправление опечаток → YandexGPT → Ответ")
        else:
            print("\n❌ Ошибка при обращении к YandexGPT")
    else:
        print("\n❌ Тест транскрипции не прошел")

if __name__ == "__main__":
    main()
