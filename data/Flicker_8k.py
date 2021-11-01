import pickle5 as pickle
from utils import *
from .TokenizeData import Tokenize


class Flicker8k:
    def __init__(self, img_pth, txt_pth, cap_file, img_name, cfg, d_limiter=40000):

        super().__init__()
        self.img_pth = img_pth
        self.txt_pth = txt_pth
        self.cap_file = cap_file
        self.img_name = img_name

        self.d_limiter = d_limiter
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

    def get_data(self):

        all_caps = []
        img_name = []

        data = pre_process(self.img_pth, self.txt_pth)
        with open(self.cap_file, 'wb') as file:
            for caption in data['captions'].astype(str):
                caption = '<start> ' + caption + ' <end>'
                all_caps.append(caption)

            pickle.dump(all_caps, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.img_name, 'wb') as file:
            for ann in data['filename']:
                full_image_path = self.img_pth + ann
                img_name.append(full_image_path)

            pickle.dump(img_name, file, protocol=pickle.HIGHEST_PROTOCOL)

        return all_caps, img_name
