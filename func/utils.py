from googletrans import Translator
import collections
import random
import textwrap


def percentage_to_100(current_percentage):
    return round(100 - current_percentage, 2)

def rephrase_text(text):
    return text  

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='en', dest='ru')
    return translation.text

def makarov_chain_rewrite(text):
    words = text.split()
    chain_length = 3  

    possibles = collections.defaultdict(list)
    for i in range(len(words) - chain_length):
        context = tuple(words[i:i+chain_length])
        next_word = words[i+chain_length]
        possibles[context].append(next_word)


    output = []
    current_context = random.choice(list(possibles.keys()))
    output.extend(current_context)

    while len(output) < 100:
        next_word = random.choice(possibles.get(current_context, ['']))
        if next_word == '':
            break
        output.append(next_word)
        current_context = tuple(output[-chain_length:])

    return textwrap.fill(' '.join(output))