import os
import os.path as op
import argparse
import time
import json
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report, precision_score,  recall_score, f1_score, accuracy_score
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification


def design_prompt():
    # task-specific
    ret_dict = {
        'main': {
            'template': '{"placeholder":"text_b"}是对{"placeholder":"text_a"}的讽刺吗？{"mask"}',
            'label_words': [['否'], ['是']],
        },
    }

    return ret_dict


def get_dataloader(raw_data, tokenizer, promptTemplate, WrapperClass, batch_size=32, max_len=256, split='train', sample:int=None):
    dataset = []
    for i, example in enumerate(raw_data):
        dataset.append(
            InputExample(
                guid=i,
                text_a=example['topic'],
                text_b=example['text'],
                label=int(example['origin_iron'])
            )
        )

    if split == 'train':
        return PromptDataLoader(
            dataset=dataset[:sample] if sample else dataset,
            tokenizer=tokenizer,
            template=promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=batch_size,
            max_seq_length=max_len,
            shuffle=True,
        )
    else:
        return PromptDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            template=promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=batch_size,
            max_seq_length=max_len,
        )


def run(args, seed=42):
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

    plm, tokenizer, model_config, WrapperClass = load_plm('bert', 'bert-base-chinese')

    # define tempalte and label_words
    prompt_dict = design_prompt()
    print('== template: %s' % prompt_dict['main']['template'])
    print('== label words: ', prompt_dict['main']['label_words'])

    promptTemplate = ManualTemplate(
        text=prompt_dict['main']['template'],
        tokenizer=tokenizer,
    )

    promptVerbalizer = ManualVerbalizer(
        label_words=prompt_dict['main']['label_words'],
        tokenizer=tokenizer,
    )

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
        freeze_plm=args.freeze_plm,
    )
    promptModel = promptModel.to(device)

    if args.do_train:
        train_dataloader = get_dataloader(train_data, tokenizer, promptTemplate, WrapperClass, sample=args.num_sample,
                                          batch_size=args.batch_size, max_len=args.max_length, split='train')
        valid_dataloader = get_dataloader(valid_data, tokenizer, promptTemplate, WrapperClass,
                                          batch_size=args.batch_size, max_len=args.max_length, split='valid')

        loss_func = nn.CrossEntropyLoss()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        total_steps = len(train_dataloader) * args.epoch
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                    num_training_steps=total_steps)

        # train
        best_f1 = 0.0
        for epoch in tqdm(range(args.epoch), desc='Training'):
            tic = time.time()

            train_loss = 0.0
            train_size = 0
            promptModel.train()
            optimizer.zero_grad()
            for batch in tqdm(train_dataloader, desc='Epoch: %s' % epoch):
                batch = batch.to(device)
                logits = promptModel(batch)
                loss = loss_func(logits, batch['label'])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                batch_size = len(batch['label'])
                train_loss += loss.item() * batch_size
                train_size += batch_size

            train_loss = train_loss / train_size

            # validation
            y_pred = []
            y_true = []
            promptModel.eval()
            with torch.no_grad():
                for batch in tqdm(valid_dataloader):
                    batch = batch.to(device)
                    logits = promptModel(batch)
                    y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
                    y_true.append(batch["label"].cpu().numpy())

            y_pred = np.concatenate(y_pred, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            results = classification_report(y_pred=y_pred, y_true=y_true, digits=4)
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

            # select best model by f1 score on validation set
            if best_f1 < f1:
                best_f1 = f1
                print('-- EPOCH %s: new best model! save to %s' % (epoch, model_save_path))
                torch.save(promptModel.state_dict(), model_save_path)

    if args.do_eval:
        # test
        promptModel.load_state_dict(torch.load(model_save_path))
        test_dataloader = get_dataloader(test_data, tokenizer, promptTemplate, WrapperClass,
                                         batch_size=args.batch_size, max_len=args.max_length, split='test')

        guids = []
        y_pred = []
        y_true = []
        promptModel.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch = batch.to(device)
                logits = promptModel(batch)
                y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
                y_true.append(batch["label"].cpu().numpy())
                guids.append(batch['guid'].cpu().numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        results = classification_report(y_pred=y_pred, y_true=y_true, digits=4)
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

        guids = np.concatenate(guids, axis=0)
        pd.DataFrame({'id': guids, 'pred': y_pred, 'true': y_true}).to_csv(
            os.path.join(args.save_path, 'test_results_%s.csv' % seed), index=False)

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
    parser.add_argument('--do_train', action="store_true", default=False)
    parser.add_argument('--do_eval', action="store_true", default=False)
    parser.add_argument('--epoch', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=2e-3)
    parser.add_argument('--num_sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--save_path', type=str, default='./save/%s/' % time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
    parser.add_argument('--freeze_plm', action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    multiple_results = []
    for seed in tqdm(args.seeds, desc='Multiple validation...'):
        print('== Hyper-parameters: \n'
              '-- Epoch: %s\n'
              '-- Batch Size: %s\n'
              '-- Max Length: %s\n'
              '-- Learning Rate: %s\n'
              '-- Weight Decay: %s\n'
              '-- Number of Sampling: %s\n'
              '-- Freeze Pre-trained LM: %s\n'
              '-- Seed： %s\n' % (args.epoch, args.batch_size, args.max_length, args.lr, args.weight_decay,
                                 args.num_sample, args.freeze_plm, seed))
        res_dict = run(args, seed)
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

