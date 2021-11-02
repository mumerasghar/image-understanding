import json
import warnings
import numpy as np
from data import CreateDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

results = {}
dict2 = dict()
FULL_PATH = "./Dataset/COCO/extracted/val2014/"


def read_file(path):
    karpathy_test = './Dataset/COCO/splits/finalkarpathysplit_test.json'

    f = open(karpathy_test)
    dict1 = json.load(f)

    for keys in dict1:
        *_, _t = keys['dir'].split('/')
        dict2[keys['image_id']] = path + _t


def append_to_list(id, name):
    word = tokenizer.encoder.decode([int(id[0])])
    if name in results.keys():
        if results[name][len(results[name]) - 1] != '<end>':
            results[name].append(word)
    else:
        results[name] = [word]

    return results[name]


def i_map_func(img_name):
    a = dict2.get(img_name)
    *_, temp = a.split('/')
    f_img_tensor = np.load(f'{FULL_PATH}{temp}.npy')
    img_tensor = np.load(a + '.npy')
    return img_tensor, img_name, f_img_tensor


def evaluate(image, names, tokenize, transformer, device):
    global tokenizer
    tokenizer = tokenize
    start_token = tokenizer.encoder.encode("<start>")
    end_token = tokenizer.encoder.encode("<end>")
    decoder_input = [start_token]
    decoder_input = np.repeat(decoder_input, repeats=image.shape[0])
    output = torch.from_numpy(np.expand_dims(decoder_input, axis=-1)) # tokens

    output = output.to(device)
    image = image.to(device)
    end_token = end_token.to(device)
    for i in range(40):
        predictions = transformer(image, output)
        predictions = predictions[:, -1:, :]
        predicted_id = torch.argmax(predictions, -1)

        if torch.all(predicted_id == end_token):
            break
        l_name = names.detach().numpy()
        l_predicted_id = predicted_id.cpu().detach().numpy()

        l = list(map(append_to_list, l_predicted_id, l_name))
        output = torch.cat((output, predicted_id), -1)


class InferenceDataset(Dataset):
    def __init__(self, _img_name_vector):
        self.img_name_vector = _img_name_vector

    def _load_features(self, img_name):
        a = dict2.get(img_name)
        *_, temp = a.split('/')
        img_tensor = np.load(a + '.npy')
        return img_tensor, img_name

    def __len__(self):
        return len(self.img_name_vector)

    def __getitem__(self, item):
        return self._load_features(self.img_name_vector[item])


def karpathy_inference(tokenizer, transformer,device, cfg):
    read_file(cfg['k_INFERENCE'])
    l = (dict2.keys())
    l = list(set(l))

    i_data = DataLoader(InferenceDataset(l), 128)

    for batch_idx, (image, name) in enumerate(i_data):
        print(f'Karpathy split inference : {batch_idx}')
        evaluate(image, name, tokenizer, transformer,device)

    finallist = []
    for i in results.keys():
        imagecap = results.get(i)
        result_join = ' '.join(imagecap)
        result_join = result_join[:(len(result_join) - 6)]
        finallist.append({'image_id': int(i), 'caption': result_join})

    jsonString = json.dumps(finallist)
    jsonFile = open("./captions_val2014_result_results.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()


if __name__ == '__main__':
    import yaml

    DATASET = 'COCO_RCNN'
    with open('../cfg/cfg.yaml', 'r') as f:
        cfg = yaml.load(f)
        cfg = cfg[DATASET]
    karpathy_inference(None, None, cfg)
