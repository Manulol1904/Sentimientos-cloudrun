import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúñ\s]', '', text)
    return text