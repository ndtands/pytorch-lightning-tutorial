import re
import unicodedata
from typing import List


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def preprocess_text(text: str):
    def strip_emoji(text):
        RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')

        return RE_EMOJI.sub(r'', text)

    def pad_white_space(text):
        text = re.sub("""(?<! )(?=[%!?()'"])|(?<=[%!?()'"])(?! )""", r' ', text)

        return text

    text = unicodedata.normalize('NFC', text)
    text = text.replace('\n', ' ')
    text = strip_emoji(text)
    text = pad_white_space(text)

    words = text.split(' ')
    words = [word for word in words if word != '']

    new_words = []
    for word in words:
        if len(word) > 1 and word[-1] in [',', '.', '...']:
            new_words.append(word[:-1])
            new_words.append(word[-1])
        else:
            new_words.append(word)

    normed_text = ' '.join(new_words)

    return normed_text
