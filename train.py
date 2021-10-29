from data import *

with open('./config/config.yml', 'r') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

data = Flicker8k(_cfg['DATASET_DIR'], _cfg['CAP_FILE'], _cfg['IMG_NAME'])
d_loader = DataLoader(data, _cfg['BATCH_SIZE'], shuffle=_cfg['SHUFFLE'])

for idx, (batch) in enumerate(d_loader):
    print(batch[1].shape)
    break
