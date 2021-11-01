import os
import json
import pandas as pd
import pickle5 as pickle
from utils import *
from .TokenizeData import Tokenize


class COCO:
    def __init__(self, img_pth, txt_pth, cap_file, img_name, cfg, d_limiter=40000):

        super().__init__()
        self.img_pth = img_pth
        self.txt_pth = txt_pth
        self.cap_file = cap_file
        self.img_name = img_name

        self.d_limiter = d_limiter
        self.tokenizer = lambda s: s.split()
        self.max_length = self._read_data()

    def _read_data(self):
        if os.path.isfile(self.cap_file) and os.path.isfile(self.img_name):
            print("[+] found cached caption.pickle.")
            with open(self.cap_file, 'rb') as f:
                train_captions = pickle.load(f)
                self.train_captions = train_captions[:self.d_limiter]

            print('[+] found cached img_name.pickle.')
            with open(self.img_name, 'rb') as f:
                img_name_vector = pickle.load(f)
                self.img_name_vector = img_name_vector[:self.d_limiter]

        else:
            print('[+] creating cached caption.pickle and img_name.pickle.')
            train_captions, img_name_vector = self.get_data()

            self.train_captions = train_captions[:self.d_limiter]
            self.img_name_vector = img_name_vector[:self.d_limiter]

        return max_length(self.train_captions)

    @staticmethod
    def copy_sub_data(a):
        ''' Unnecessary Code '''
        import shutil
        _src = './Dataset/COCO/extracted/train2014/'
        _dest = './Dataset/COCO/extracted/coco_training/'

        a = a[0:40000]
        _a_item = set(a)
        for i in _a_item:
            _img = f"COCO_train2014_{''.join(['0' for i in range(12 - len(str(i)))])}{i}"
            shutil.copy(f'{_src}{_img}.jpg', _dest)
            shutil.copy(f'{_src}{_img}.jpg.npy', _dest)

    def pre_process(self):
        with open(self.txt_pth) as file:
            annotations = json.load(file)
            annotations = annotations['annotations']

        df = []
        for item in annotations:
            t = (item['image_id'], item['caption'].lower())
            df.append(t)

        df = pd.DataFrame(df, columns=['filename', 'captions'])

        return df

    def get_data(self):

        all_caps = []
        img_name = []

        df = self.pre_process()

        print('\t[+] formatting captions')
        with open(self.cap_file, 'wb') as file:
            for cap in df['captions'].astype(str):
                cap = '<start> ' + cap + ' <end>'
                all_caps.append(cap)
            pickle.dump(all_caps, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.img_name, 'wb') as file:
            for f_name in df['filename']:
                _name = ''.join(['0' for _ in range(12 - len(str(f_name)))])
                c_adr = f"COCO_train2014_{_name}{f_name}.jpg"
                img_name.append(self.img_pth + c_adr)

            pickle.dump(img_name, file, protocol=pickle.HIGHEST_PROTOCOL)

        return all_caps, img_name
