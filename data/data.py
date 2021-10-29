import os
import yaml
import numpy as np
import pickle5 as pickle
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders.text import StaticTokenizerEncoder, pad_tensor


class Flicker8k(Dataset):
    def __init__(self, dir_path, cap_file, all_img_name, max_length=40, d_limiter=40000):

        self.dir_path = dir_path
        self.d_limiter = d_limiter
        self.tokenizer = lambda s: s.split()
        self.max_length = max_length
        self._read_data(cap_file, all_img_name)
        self.vocab_size = self.encoder.vocab_size

    def _read_data(self, cap_file, all_img_name):
        if not os.path.isfile(cap_file):
            print(f'{cap_file}.pickle not Found.')
            raise NotImplemented
        else:
            with open(cap_file, 'rb') as f:
                train_captions = pickle.load(f)
                self.train_captions = train_captions[:self.d_limiter]

        if not os.path.isfile(all_img_name):
            print(f'{all_img_name}.pickle not Found.')
            raise NotImplemented
        else:
            with open(all_img_name, 'rb') as f:
                img_name_vector = pickle.load(f)
                self.img_name_vector = img_name_vector[:self.d_limiter]

        self._encode_data(train_captions)

    def _encode_data(self, train_captions):
        self.encoder = StaticTokenizerEncoder(train_captions,
                                              tokenize=self.tokenizer,
                                              min_occurrences=2,
                                              padding_index=0,
                                              unknown_index=1)

        encoded_data = [self.encoder.encode(e) for e in train_captions]
        self.encoded_data = [pad_tensor(x, length=self.max_length) for x in encoded_data]



    def _load_features(self, cap_name):
        img_tensor = np.load(cap_name + '.npy')
        return img_tensor

    def __len__(self):
        return len(self.img_name_vector)

    def __getitem__(self, item):
        return self._load_features(self.img_name_vector[item]), self.encoded_data[item]



