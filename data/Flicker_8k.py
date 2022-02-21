import os
import string

import pandas as pd
import pickle5 as pickle

from .TokenizeData import Tokenize


def remove_numeric(text):
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if isalpha:
            text_no_numeric += " " + word
    return text_no_numeric


def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word

    return text_len_more_than1


def remove_punctuation(text_original):
    translator = str.maketrans(
        string.punctuation, " " * len(string.punctuation)
    )  # map punctuation to space
    return text_original.translate(translator)


def split_text(text):
    data_txt = []
    for line in text.split("\n"):
        col = line.split("\t")
        if len(col) == 1:
            continue
        w = col[0].split("#")
        data_txt.append(w + [col[1].lower()])

    return data_txt


def text_clean(text_original):
    text = remove_punctuation(text_original)
    # text = remove_single_character(text)
    text = remove_numeric(text)

    return text


def pre_process(img_pth, txt_pth):
    jpgs = os.listdir(img_pth)
    print(f"  [+] Total image in dataset is {len(jpgs)}.")

    with open(txt_pth, "r") as file:
        text = file.read()

    s_text = split_text(text)

    df = pd.DataFrame(s_text, columns=["filename", "index", "captions"])
    df = df.reindex(columns=["index", "filename", "captions"])
    df = df[df.filename != "2258277193_586949ec62.jpg.1"]

    vocabulary = []
    for txt in df.captions.values:
        vocabulary.extend(txt.split())

    print(f"  [+] Vocabulary Size {len(set(vocabulary))}.")

    for i, cap in enumerate(df.captions.values):
        new_cap = text_clean(cap)
        df["captions"].iloc[i] = new_cap

    return df


class Flicker8k(Tokenize):
    def __init__(self, img_pth, txt_pth, cap_file, img_name, cfg, d_limiter=40000):

        self.img_pth = img_pth
        self.txt_pth = txt_pth
        self.cap_file = cap_file
        self.img_name = img_name

        self.d_limiter = d_limiter
        self.tokenizer = lambda s: s.split()
        self._read_data()

        super().__init__(
            self.img_name_vector,
            self.train_captions,
            self.tokenizer,
            self.max_length,
            cfg,
        )

    def _read_data(self):
        if os.path.isfile(self.cap_file) and os.path.isfile(self.img_name):
            print("[+] found cached caption.pickle.")
            with open(self.cap_file, "rb") as f:
                train_captions = pickle.load(f)
                self.train_captions = train_captions[: self.d_limiter]

            print("[+] found cached img_name.pickle.")
            with open(self.img_name, "rb") as f:
                img_name_vector = pickle.load(f)
                self.img_name_vector = img_name_vector[: self.d_limiter]

        else:
            print("[+] creating cached caption.pickle and img_name.pickle.")
            train_captions, img_name_vector = self.get_data()

            self.train_captions = train_captions[: self.d_limiter]
            self.img_name_vector = img_name_vector[: self.d_limiter]

        self.max_length = len(max(self.train_captions, key=len).split(" "))

    def get_data(self):

        all_caps = []
        img_name = []

        data = pre_process(self.img_pth, self.txt_pth)
        with open(self.cap_file, "wb") as file:
            for caption in data["captions"].astype(str):
                caption = "<start> " + caption + " <end>"
                all_caps.append(caption)

            pickle.dump(all_caps, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.img_name, "wb") as file:
            for ann in data["filename"]:
                full_image_path = self.img_pth + ann
                img_name.append(full_image_path)

            pickle.dump(img_name, file, protocol=pickle.HIGHEST_PROTOCOL)

        return all_caps, img_name
