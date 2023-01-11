import re
import unicodedata
import typing as t


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

def split_content(lines: t.List)  -> t.List:
    out = []
    temp = []
    for line in lines:
        if line.strip()=='':
            out.append(temp)
            temp = []
        else:
            temp.append(line)
    if temp != []:
        out.append(temp)
    return out

def check(lines: t.List) -> bool:
    for line in lines:
        if line.strip()[0] in ['-','•']:
            return True
    return False

def merger(cluster: t.List) -> t.List:
    out = []
    temp =[]
    for line in cluster:
        if line.strip()[0] in ['-','•']:
            out.append("".join(temp))
            temp = [line]
        else:
            temp.append(line)
    if temp != []:
        out.append("".join(temp))
    return out

def preprocess_JD(JD: str) -> t.List:
    clusters = split_content(JD.split('\n'))
    out = []
    for cluster in clusters :
        if check(cluster):
            out.extend(merger(cluster))
        else:
            out.append(''.join(cluster))
    return out