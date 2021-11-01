import os
import string
import pandas as pd


def remove_numeric(text):
    text_no_numeric = ''
    for word in text.split():
        isalpha = word.isalpha()
        if isalpha:
            text_no_numeric += ' ' + word
    return text_no_numeric


def text_clean(text_original):
    text = remove_punctuation(text_original)
    # text = remove_single_character(text)
    text = remove_numeric(text)

    return text


def remove_single_character(text):
    text_len_more_than1 = ''
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += ' ' + word

    return text_len_more_than1


def remove_punctuation(text_original):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    return text_original.translate(translator)


def split_text(text):
    data_txt = []
    for line in text.split('\n'):
        col = line.split('\t')
        if len(col) == 1:
            continue
        w = col[0].split('#')
        data_txt.append(w + [col[1].lower()])

    return data_txt


def pre_process(img_pth, txt_pth):
    jpgs = os.listdir(img_pth)
    print(f'  [+] Total image in dataset is {len(jpgs)}.')

    with open(txt_pth, 'r') as file:
        text = file.read()

    s_text = split_text(text)

    df = pd.DataFrame(s_text, columns=['filename', 'index', 'captions'])
    df = df.reindex(columns=['index', 'filename', 'captions'])
    df = df[df.filename != '2258277193_586949ec62.jpg.1']

    vocabulary = []
    for txt in df.captions.values:
        vocabulary.extend(txt.split())

    print(f'  [+] Vocabulary Size {len(set(vocabulary))}.')

    for i, cap in enumerate(df.captions.values):
        new_cap = text_clean(cap)
        df['captions'].iloc[i] = new_cap

    return df


def max_length(_captions):
    _max = 0
    for i in _captions:
        _len = len(i.split(' '))
        if _len > _max:
            _max = _len
    return _max