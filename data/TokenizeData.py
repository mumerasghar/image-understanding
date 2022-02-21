import numpy as np
from torch.utils.data import Dataset
from torchnlp.encoders.text import StaticTokenizerEncoder, pad_tensor


class Tokenize(Dataset):
    def __init__(
        self, _img_name_vector, _train_captions, _tokenizer, _max_length, _cfg
    ):
        self.img_name_vector = _img_name_vector
        self.tokenizer = _tokenizer
        self.max_length = _max_length
        self.min_occur = _cfg["MIN_OCCURRENCES"]
        self._encode_data(_train_captions)

    def _encode_data(self, train_captions):
        self.encoder = StaticTokenizerEncoder(
            train_captions,
            tokenize=self.tokenizer,
            min_occurrences=self.min_occur,
            padding_index=0,
            unknown_index=1,
        )

        encoded_data = [self.encoder.encode(e) for e in train_captions]
        self.encoded_data = [
            pad_tensor(x, length=self.max_length) for x in encoded_data
        ]
        self.vocab_size = self.encoder.vocab_size

    def _load_features(self, cap_name):
        img_tensor = np.load(cap_name + ".npy")
        return img_tensor

    def __len__(self):
        return len(self.img_name_vector)

    def __getitem__(self, item):
        return (
            self._load_features(self.img_name_vector[item]),
            self.encoded_data[item],
            self.img_name_vector[item],
        )
