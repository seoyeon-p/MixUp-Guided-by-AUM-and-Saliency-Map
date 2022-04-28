import argparse
import csv
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW, AutoModel, AutoTokenizer
from tqdm import tqdm
from aum import AUMCalculator
from multiprocessing import Process, Pool
from torch.multiprocessing import Pool, Process, set_start_method
import os



csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
parser.add_argument('--task', type=str, help='task name (SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG)')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--label_smoothing', type=float, default=-1., help='label smoothing \\alpha')
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--train_from', type=str)
parser.add_argument('--multigpu',action='store_true')
args = parser.parse_args()
print(args)

save_dir = './output/'+args.task
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
aum_calculator = AUMCalculator(save_dir,compressed=False)

if args.multigpu:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    args.device = torch.device("cuda")

assert args.task in ('SNLI', 'MNLI', 'QQP', 'TwitterPPDB', 'SWAG', 'HellaSWAG')
assert args.model in ('bert-base-uncased', 'roberta-base')

if args.task in ('SNLI', 'MNLI'):
    n_classes = 3
elif args.task in ('QQP', 'TwitterPPDB'):
    n_classes = 2
elif args.task in ('SWAG', 'HellaSWAG'):
    n_classes = 1


def cuda(tensor):
    """Places tensor on CUDA device."""
    return tensor.to(args.device)


def load(dataset, batch_size, shuffle):
    """Creates data loader with dataset and iterator options."""
    print(len(dataset))
    return DataLoader(dataset, batch_size, shuffle=shuffle)


def adamw_params(model):
    """Prepares pre-trained model parameters for AdamW optimizer."""

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return params


def encode_pair_inputs(sentence1, sentence2, passed_tokenizer):
    """
    Encodes pair inputs for pre-trained models using the template
    [CLS] sentence1 [SEP] sentence2 [SEP]. Used for SNLI, MNLI, QQP, and TwitterPPDB.
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = passed_tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=args.max_seq_length
    )
    input_ids = inputs['input_ids']
    if args.model == 'bert-base-uncased':
        segment_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [passed_tokenizer.pad_token_id] * padding_length
    if args.model == 'bert-base-uncased':
        segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    if args.model == 'bert-base-uncased':
        for input_elem in (input_ids, segment_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        return (
            cuda(torch.tensor(input_ids)).long(),
            cuda(torch.tensor(segment_ids)).long(),
            cuda(torch.tensor(attention_mask)).long(),
        )
    else:
        for input_elem in (input_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        return (
            cuda(torch.tensor(input_ids)).long(),
            cuda(torch.tensor(attention_mask)).long(),
        )



def encode_mc_inputs(context, start_ending, endings, tokenizer):
    """
    Encodes multiple choice inputs for pre-trained models using the template
    [CLS] context [SEP] ending_i [SEP] where 0 <= i < len(endings). Used for
    SWAG and HellaSWAG. Returns input_ids, segment_ids, and attention_masks.
    """
    context_tokens = tokenizer.tokenize(context)
    start_ending_tokens = tokenizer.tokenize(start_ending)
    all_input_ids = []
    all_segment_ids = []
    all_attention_masks = []
    for ending in endings:
        ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
        inputs = tokenizer.encode_plus(
            " ".join(context_tokens), " ".join(ending_tokens), add_special_tokens=True, max_length=args.max_seq_length
        )
        input_ids = inputs['input_ids']
        if args.model == 'bert-base-uncased':
            segment_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        padding_length = args.max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        if args.model == 'bert-base-uncased':
            segment_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        if args.model == 'bert-base-uncased':
            for input_elem in (input_ids, segment_ids, attention_mask):
                assert len(input_elem) == args.max_seq_length
            all_input_ids.append(input_ids)
            all_segment_ids.append(segment_ids)
            all_attention_masks.append(attention_mask)
        else:
            for input_elem in (input_ids, attention_mask):
                assert len(input_elem) == args.max_seq_length
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
    if args.model == 'bert-base-uncased':
        return (
            cuda(torch.tensor(all_input_ids)).long(),
            cuda(torch.tensor(all_segment_ids)).long(),
            cuda(torch.tensor(all_attention_masks)).long(),
        )
    else:
        return (
            cuda(torch.tensor(all_input_ids)).long(),
            cuda(torch.tensor(all_attention_masks)).long(),
        )

def encode_label(label):
    """Wraps label in tensor."""

    return cuda(torch.tensor(label)).long()


class SNLIProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[1]
                    sentence1 = row[4]
                    sentence2 = row[7]
                    label = row[2]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples


class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label))
                except:
                    pass
        return samples


class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[0]
                    sentence1 = row[4]
                    sentence2 = row[5]
                    label = row[1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples


class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples.append((sentence1, sentence2, label))
                except:
                    pass
        return samples


class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    context = str(row[4])
                    start_ending = str(row[5])
                    endings = row[7:11]
                    label = int(row[-1])
                    samples.append((context, start_ending, endings, label))
                except:
                    pass
        return samples


class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path):
        samples = []
        with open(path) as f:
            desc = f'loading \'{path}\''
            for line in f:
                try:
                    line = line.rstrip()
                    input_dict = json.loads(line)
                    context = input_dict['ctx_a']
                    start_ending = input_dict['ctx_b']
                    endings = input_dict['endings']
                    label = input_dict['label']
                    samples.append((context, start_ending, endings, label))
                except:
                    pass
        return samples


def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor']()




class TextDataset(Dataset):
    """
    Task-specific dataset wrapper. Used for storing, retrieving, encoding,
    caching, and batching samples.
    """

    def __init__(self, path, processor,passed_tokenizer):
        self.samples = processor.load_samples(path)
        self.cache = {}
        self.tokenizer = passed_tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]
            if args.task in ('SNLI', 'MNLI', 'QQP', 'MRPC', 'TwitterPPDB'):
                sentence1, sentence2, label, guid = sample
                if args.model == 'bert-base-uncased':
                    input_ids, segment_ids, attention_mask = encode_pair_inputs(
                        sentence1, sentence2, self.tokenizer
                    )
                else:
                    input_ids, attention_mask = encode_pair_inputs(
                        sentence1, sentence2, self.tokenizer
                    )
                packed_inputs = (sentence1, sentence2)
            elif args.task in ('SWAG', 'HellaSWAG'):
                if args.model == 'bert-base-uncased':
                    context, ending_start, endings, label = sample
                    input_ids, segment_ids, attention_mask = encode_mc_inputs(
                        context, ending_start, endings, self.tokenizer
                    )
                else:
                    context, ending_start, endings, label = sample
                    input_ids, attention_mask = encode_mc_inputs(
                        context, ending_start, endings, self.tokenizer
                    )
            label_id = encode_label(label)
            if args.model == 'bert-base-uncased':
                res = ((input_ids, segment_ids, attention_mask), label_id, guid)
            else:
                res = ((input_ids, attention_mask), label_id, guid)
            self.cache[i] = res
        return res


class Model(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, segment_ids, attention_mask):
        # On SWAG and HellaSWAG, collapse the batch size and
        # choice size dimension to process everything at once
        if args.task in ('SWAG', 'HellaSWAG'):
            n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            if args.model == 'bert-base-uncased':
                segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        transformer_params = {
            'input_ids': input_ids,
            'token_type_ids': (
                segment_ids if args.model == 'bert-base-uncased' else None
            ),
            'attention_mask': attention_mask,
        }

        transformer_outputs = self.model(**transformer_params)
        if args.task in ('SWAG', 'HellaSWAG'):
            pooled_output = transformer_outputs[1]
            logits = self.classifier(pooled_output)
            logits = logits.view(-1, n_choices)
        else:
            cls_output = transformer_outputs[0][:, 0]
            logits = self.classifier(cls_output)
        return logits


def label_smoothing(label,classes, smoothing_value, confidence):
    neg_one_hot = cuda(torch.full((classes,), smoothing_value))
    neg_label_one_hot = neg_one_hot.repeat(label.size(0), 1)
    neg_label_convert = label.view(-1, 1)
    neg_label_one_hot.scatter_(1, neg_label_convert, confidence)
    return neg_label_one_hot



def train(dataset):
    """Fine-tunes pre-trained model on training set."""

    model.train()
    train_loss = 0.
    train_loader = tqdm(load(dataset, args.batch_size, True))
    optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
    for i, (inputs,label,guid) in enumerate(train_loader,1):
        optimizer.zero_grad()
        if args.model == 'bert-base-uncased':
            logits = model(*inputs)
        else:
            logits = model(inputs[0],None,inputs[1])
        logits.retain_grad()
        loss = criterion(logits,label)
        loss.backward()
        records = aum_calculator.update(logits, label, guid, logits.grad.data.abs().tolist())
        train_loss += loss.item()
        train_loader.set_description(f'train loss = {(train_loss / i):.6f}')
        if args.max_grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    return train_loss / len(train_loader)


def evaluate(dataset):
    """Evaluates pre-trained model on development set."""

    model.eval()
    eval_loss = 0.
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    for i, (inputs, label, guid) in enumerate(eval_loader, 1):
        with torch.no_grad():
            if args.model == 'bert-base-uncased':
                loss = criterion(model(*inputs), label)
            else:
                loss = criterion(model(inputs[0],None,inputs[0]),label)
        eval_loss += loss.item()
        eval_loader.set_description(f'eval loss = {(eval_loss / i):.6f}')
    return eval_loss / len(eval_loader)

if args.multigpu:
    model = Model()
    model = nn.DataParallel(model).to(args.device)
else:
    model = cuda(Model())
if args.train_from:
    model.load_state_dict(torch.load(args.train_from))
processor = select_processor()
tokenizer = AutoTokenizer.from_pretrained(args.model)

if args.label_smoothing == -1:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = LabelSmoothingLoss(args.label_smoothing)

if args.train_path:
    train_dataset = TextDataset(args.train_path, processor,tokenizer)
    print(f'train samples = {len(train_dataset)}')
    #train_dataset = DatasetWithIndex(train_dataset)
if args.dev_path:
    dev_dataset = TextDataset(args.dev_path, processor,tokenizer)
    print(f'dev samples = {len(dev_dataset)}')
if args.test_path:
    test_dataset = TextDataset(args.test_path, processor,tokenizer)
    print(f'test samples = {len(test_dataset)}')
    #test_dataset =  DatasetWithIndex(test_dataset)

if args.do_train:
    print('*** training ***')
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(train_dataset)
        eval_loss = evaluate(dev_dataset)
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), args.ckpt_path)
        print(
            f'epoch = {epoch} | '
            f'train loss = {train_loss:.6f} | '
            f'eval loss = {eval_loss:.6f}'
        )
        save_path = args.ckpt_path + "_"+ str(epoch)
        torch.save(model.state_dict(), save_path)

    aum_calculator.finalize()

if args.do_evaluate:
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError(f'\'{args.ckpt_path}\' does not exist')

    print()
    print('*** evaluating ***')

    output_dicts = []
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    test_loader = tqdm(load(test_dataset, args.batch_size, False))

    for i, (inputs, label, sample_ids) in enumerate(test_loader,1):
        with torch.no_grad():
            if args.model =='bert-base-uncased':
                logits = model(*inputs)
            else:
                logits = model(inputs[0],None,inputs[1])

            for j in range(logits.size(0)):
                probs = F.softmax(logits[j], -1)
                output_dict = {
                    'index': args.batch_size * i + j,
                    'true': label[j].item(),
                    'pred': logits[j].argmax().item(),
                    'conf': probs.max().item(),
                    'logits': logits[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)

    print(f'writing outputs to \'{args.output_path}\'')
    with open(args.output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')

    y_true = [output_dict['true'] for output_dict in output_dicts]
    y_pred = [output_dict['pred'] for output_dict in output_dicts]
    y_conf = [output_dict['conf'] for output_dict in output_dicts]

    accuracy = accuracy_score(y_true, y_pred) * 100.
    f1 = f1_score(y_true, y_pred, average='macro') * 100.
    confidence = np.mean(y_conf) * 100.

    results_dict = {
        'accuracy': accuracy_score(y_true, y_pred) * 100.,
        'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
        'confidence': np.mean(y_conf) * 100.,
    }
    for k, v in results_dict.items():
        print(f'{k} = {v}')
