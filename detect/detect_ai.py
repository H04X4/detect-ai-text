from transformers import pipeline
from googletrans import Translator


pipe = pipeline("text-classification", model="tommyliphys/ai-detector-distilbert")

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='auto', dest='en')
    return translation.text


def classify_text(input_text):

    if not any(char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' for char in input_text):
    
        translated_text = translate_to_english(input_text)
    else:
        translated_text = input_text

    results = pipe(translated_text)

    classification_results = []
    for result in results:
        label = result['label']
        score = result['score']
        score_percent = round(score * 100, 2)
        classification_results.append({"label": label, "score": score_percent})
    
    return classification_results
