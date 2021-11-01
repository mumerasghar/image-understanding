import math
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models import Transformer, ScheduledOptimizer
from data import Data


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    if smoothing:
        gold = gold.contiguous().view(-1)
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='mean')
    return loss


def train_epoch(model, training_data, optimizer, device):
    """ Epoch operation in training phase"""
    model.train()
    total_loss, train_accuracy = 0, 0

    desc = '  - (Training)   '
    tq = tqdm(training_data, mininterval=2, desc=desc, leave=False)
    for img_tensor, target, img_name in tq:
        # prepare data
        src_seq = img_tensor.to(device)
        target_inp = target[:, :-1].contiguous().to(device)
        target_real = target[:, 1:].contiguous().to(device)

        # forward prop
        optimizer.zero_grad()

        pred = model(src_seq, target_inp)
        loss = cal_loss(pred.permute((0, 2, 1)), target_real, 0)
        loss.backward()
        optimizer.step_and_update_lr()

        tq.set_description(f'Loss {loss.item()}')
        total_loss += loss.item()

    return total_loss, train_accuracy


def eval_epoch(model, data, tokenizer, device):
    """ Epoch operation in evaluation phase """

    model.eval()
    total_loss = 0
    with torch.no_grad():
        img_tensor, target, img_name = next(iter(data))
        src_seq = img_tensor.to(device)

        target_inp = target[:, :-1].contiguous().to(device)

        pred = model(src_seq, target_inp)
        seq_output = torch.argmax(pred, -1)
        seq_output = seq_output.view(-1)

        with open("result.txt", "a") as f:
            f.write(f"{tokenizer.encoder.decode(seq_output)}\n")

    return total_loss, 0


def print_performances(header, ppl, accu, start_time, _lr):
    elapse = (time.time() - start_time) / 60
    print(f'\t- {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f},lr: {_lr:8.5f}, elapse: {elapse:3.3f} min')


def train(model, train_data, val_data, optimizer, tokenizer, device, cfg):
    if cfg["TENSOR_BOARD"]:
        print("[+] Using Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join('.', 'tensorboard'))

    log_train_file = os.path.join(cfg["LOGS_DIR"], 'train.log')
    log_valid_file = os.path.join(cfg["LOGS_DIR"], 'valid.log')

    print(f'[+] Training performance will be written to file: {log_train_file} and {log_valid_file}')

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_losses = []
    for epoch_i in range(cfg["EPOCHS"]):
        print(f'[ Epoch {epoch_i}]')

        start = time.time()
        train_loss, train_accu = train_epoch(model, train_data, optimizer, device)
        train_ppl = math.exp(min(train_loss, 100))

        # Current learning loss
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, val_data, tokenizer, device)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]
        checkpoint = {'epoch': epoch_i, 'settings': cfg, 'model': model.state_dict()}

        if cfg["SAVE_MODE"] == 'all':
            model_name = f'model_accu_{100 * valid_accu:3.3f}.chkpt'
            torch.save(checkpoint, model_name)
        elif cfg["SAVE_MODE"] == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(cfg["SAVE_DIR"], model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write(f'{epoch_i},{train_loss: 8.5f},{train_ppl: 8.5f},{100 * train_accu:3.3f}\n')
            log_vf.write(f'{epoch_i},{valid_loss: 8.5f},{valid_ppl: 8.5f},{100 * valid_accu:3.3f}\n')

        if cfg["TENSOR_BOARD"]:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu * 100, 'val': valid_accu * 100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)


def main():
    DATASET = "FLICKER"
    with open('./config/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = {**cfg[DATASET], **cfg["PARAMS"]}

    paths = {
        "_img_path": cfg['IMG_PATH'],
        "_text_path": cfg["TXT_PATH"],
        "_cap_file": cfg["CAP_FILE"],
        "_img_name": cfg["IMG_NAME"],
        "_cfg": cfg
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = Data(**paths)
    val_data, train_data = data.get_data
    tokenizer = data.tokenize
    training_data = DataLoader(train_data, cfg['BATCH_SIZE'], shuffle=True)
    val_data = DataLoader(val_data, 1, shuffle=True)

    transformer = Transformer(
        data.tokenize.encoder.vocab_size,
        trg_pad_idx=0,
        trg_emb_prj_weight_sharing=True,
        d_k=64, d_v=64,
        d_model=512,
        dff=2048, n_layers=1,
        n_head=8,
        dropout=0.1
    ).to(device)

    optimizer = ScheduledOptimizer(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        cfg["LR_MUL"], cfg["D_MODEL"], cfg["WARMUP_STEP"])

    train(transformer, training_data, val_data, optimizer, tokenizer, device, cfg)


if __name__ == '__main__':
    main()
