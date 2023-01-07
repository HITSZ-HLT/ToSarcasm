import os
import os.path as op
import argparse
import time
import json
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BatchEncoding, BertForSequenceClassification, BertTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, precision_score,  recall_score, f1_score, accuracy_score


def get_dataloader(raw_data, tokenizer, batch_size=32, max_len=256, split='train', sample:int=None):
    def collate(batch):
        return {'guid': torch.stack([batch[i][0] for i in range(len(batch))], dim=0),
                'inputs': BatchEncoding({
                    'input_ids': torch.stack([batch[i][1] for i in range(len(batch))], dim=0),
                    'token_type_ids': torch.stack([batch[i][2] for i in range(len(batch))], dim=0),
                    'attention_mask': torch.stack([batch[i][3] for i in range(len(batch))], dim=0),
                }),
                'labels': torch.stack([batch[i][-1] for i in range(len(batch))], dim=0)
                }

    guids = []
    all_text = []
    labels = []
    for i, example in enumerate(raw_data):
        guids.append(i)
        text_a = example['topic']
        text_b = example['text']
        all_text.append(text_b + '[SEP]' + text_a)
        labels.append(int(example['label']))
    guids = torch.tensor(guids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = tokenizer(all_text[:sample] if sample else all_text,
                        max_length=max_len,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True
                        )
    dataset = TensorDataset(guids[:sample] if sample else guids,
                            dataset['input_ids'],
                            dataset['token_type_ids'],
                            dataset['attention_mask'],
                            labels[:sample] if sample else labels)

    print("%s size: %d" % (split, len(dataset)))

    if split == 'train':
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    else:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate,
        )


def run_msd(args, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.gpu_id < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%s" % args.gpu_id)
        torch.cuda.set_device(device)

    model_save_path = os.path.join(args.save_path, 'best_model.std')

    # load data
    with open(op.join(args.data_path, 'train.json'), 'r') as f:
        train_data = json.load(f)
    with open(op.join(args.data_path, 'dev.json'), 'r') as f:
        valid_data = json.load(f)
    with open(op.join(args.data_path, 'test.json'), 'r') as f:
        test_data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
    model = model.to(device)

    train_dataloader = get_dataloader(train_data, tokenizer, batch_size=args.batch_size, max_len=args.max_length, split='train', sample=args.num_sample,)
    valid_dataloader = get_dataloader(valid_data, tokenizer, batch_size=args.batch_size, max_len=args.max_length, split='valid')
    test_dataloader = get_dataloader(test_data, tokenizer, batch_size=args.batch_size, max_len=args.max_length, split='test')

    loss_func = nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dataloader) * args.epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                num_training_steps=total_steps)

    # train
    best_f1 = 0.0
    for epoch in tqdm(range(args.epoch), desc='Training'):
        tic = time.time()

        train_loss = 0.0
        train_size = 0
        model.train()
        optimizer.zero_grad()
        for batch in tqdm(train_dataloader, desc='Epoch: %s' % epoch):
            batch_inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            logits = model(**batch_inputs)['logits']
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            batch_size = len(labels)
            train_loss += loss.item() * batch_size
            train_size += batch_size

        train_loss = train_loss / train_size

        # validation
        y_pred = []
        y_true = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader):
                batch_inputs = batch['inputs'].to(device)
                labels = batch['labels']
                logits = model(**batch_inputs)['logits']
                y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
                y_true.append(labels.cpu().numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        results = classification_report(y_pred=y_pred, y_true=y_true)
        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        macro_p = precision_score(y_true, y_pred, average='macro')
        macro_r = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        print('-- EPOCH %s: train_loss = %.4f, time: %s s\n'
              '-- EPOCH %s: valid results: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n'
              '--                   macro: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f' % (
                  epoch, train_loss, time.time() - tic, epoch, acc, p, r, f1, acc, macro_p, macro_r, macro_f1))
        print(results)
        if best_f1 < f1:
            best_f1 = f1
            print('-- EPOCH %s: new best model! save to %s' % (epoch, model_save_path))
            torch.save(model.state_dict(), model_save_path)

    # test
    model.load_state_dict(torch.load(model_save_path))

    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch_inputs = batch['inputs'].to(device)
            labels = batch['labels']
            logits = model(**batch_inputs)['logits']
            y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
            y_true.append(labels.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    results = classification_report(y_pred=y_pred, y_true=y_true)
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average='macro')
    macro_r = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    print('-- test results: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n'
          '--        macro: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f' % (acc, p, r, f1, acc, macro_p, macro_r, macro_f1))
    print(results)

    return {'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1,
            'macro precision': macro_p,
            'macro recall': macro_r,
            'macro f1': macro_f1
            }


if __name__ == '__main__':
    # hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=2e-3)
    parser.add_argument('--num_sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--save_path', type=str, default='./save/bert/')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    multiple_results = []
    for seed in tqdm(args.seeds, desc='Multiple validation...'):
        print('== Hyper-parameters: \n'
              '-- Epoch: %s\n'
              '-- Batch Size: %s\n'
              '-- Max Length: %s\n'
              '-- Learning Rate: %s\n'
              '-- Weight Decay: %s\n'
              '-- Number of Sampling: %s\n'
              '-- Seed： %s' % (
            args.epoch, args.batch_size, args.max_length, args.lr, args.weight_decay, args.num_sample, seed))
        res_dict = run_msd(args, seed)
        multiple_results.append(res_dict)

    acc = float(np.mean([r['accuracy'] for r in multiple_results]))
    p = float(np.mean([r['precision'] for r in multiple_results]))
    r = float(np.mean([r['recall'] for r in multiple_results]))
    f1 = float(np.mean([r['f1'] for r in multiple_results]))
    macro_p = float(np.mean([r['macro precision'] for r in multiple_results]))
    macro_r = float(np.mean([r['macro recall'] for r in multiple_results]))
    macro_f1 = float(np.mean([r['macro f1'] for r in multiple_results]))

    print('== averaged results: %.4f, %.4f, %.4f, %.4f\n'
          '==            macro: %.4f, %.4f, %.4f, %.4f\n' % (acc, p, r, f1, acc, macro_p, macro_r, macro_f1))

