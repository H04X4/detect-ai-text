import json
import requests


def phi3_gen(prompt):
    """Генерирует описание для изображения с использованием указанной модели."""
    
    data = {
        "model": "phi3", 
        "prompt": f"""
Действуй, как профессиональный копирайтер.
Переосмысли предоставленный текст, используя синонимы, чтобы создать уникальную версию, в которой используется естественный стиль.
Сохрани исходное форматирование и смысл текста, придерживаясь красивого литературного языка и избегая стилистических и грамматических ошибок.
Обрати внимание, что общий объем текста не должен превышать исходный больше, чем на 10%.
Предоставленный текст будет разделен на части, однако помни, что он образует единую целостность. Обеспечь высокую сложность и разнообразие слов и выражений, чтобы избежать повторов и успешно пройти проверку на уникальность. Мой текст: {prompt}.      
        """,
    }
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data)
    

    if response.status_code == 200:
        response_content = response.text.split('\n')
        description = ""
        
        for part in response_content:
            if part.strip():
                json_part = json.loads(part)
                description += json_part['response']
    
        return description