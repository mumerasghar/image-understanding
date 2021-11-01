from .TokenizeData import Tokenize
from torch.utils.data import Dataset
from .COCO import COCO
from .Flicker_8k import Flicker8k
import numpy as np
from sklearn.model_selection import train_test_split


class CreateDataset(Dataset):
    def __init__(self, _img_name_vector, _encoded_data):
        self.img_name_vector = _img_name_vector
        self.encoded_data = _encoded_data

    def _load_features(self, cap_name):
        img_tensor = np.load(cap_name + '.npy')
        return img_tensor

    def __len__(self):
        return len(self.img_name_vector)

    def __getitem__(self, item):
        return self._load_features(self.img_name_vector[item]), self.encoded_data[item], self.img_name_vector[item]


class Data(Dataset):
    def __init__(self, _img_path, _text_path, _cap_file, _img_name, _cfg):
        self.img_path = _img_path
        self.text_path = _text_path
        self.cap_file = _cap_file
        self.img_name = _img_name
        self.cfg = _cfg

    @property
    def get_data(self):
        if self.cfg["DATASET_NAME"] == 'COCO':
            f_data = COCO(self.img_path, self.text_path, self.cap_file, self.img_name, self.cfg)
        else:
            f_data = Flicker8k(self.img_path, self.text_path, self.cap_file, self.img_name, self.cfg)

        # tokenize = Tokenize(f_data.img_name_vector, f_data.train_captions, f_data.tokenizer, f_data.max_length,
        #                     self.cfg)
        self.tokenize = Tokenize(f_data.train_captions, f_data.max_length, self.cfg)
        train_name, val_name, train_cap, val_cap = train_test_split(f_data.img_name_vector, self.tokenize.encoded_data,
                                                                    test_size=0.01, random_state=0)
        train_dataset = CreateDataset(train_name, train_cap)
        val_dataset = CreateDataset(train_name, train_cap)
        return train_dataset, val_dataset
