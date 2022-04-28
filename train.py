# Taken from https://github.com/shreydesai/calibration.
import argparse
import csv
# Taken from https://github.com/shreydesai/calibration.
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
import ast



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
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--train_from', type=str, help='load checkpoint')
parser.add_argument('--multigpu',action='store_true', help='Multiple GPU')
parser.add_argument('--data_categorize',action='store_true',help='Data categorization step')
parser.add_argument('--aum_guide',action='store_true',help='enable using categorized data, which are easy-to-learn and hard-to-learn/ambiguous')
parser.add_argument('--grad_guide', action='store_true',help='enable selecting the most similar and the most dissimilar process')
parser.add_argument('--ls',action='store_true',help='enable label smoothing')
parser.add_argument('--wo_similar',action='store_true',help='without using the most similar instance in the other category')
parser.add_argument('--wo_dissimilar',action='store_true',help='without using the most dissimilar instance in the other category')
args = parser.parse_args()
print(args)

if args.data_categorize:
    save_dir = './output/'+args.task+"_"+args.model
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    aum_calculator = AUMCalculator(save_dir,compressed=False)

if args.multigpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
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

    def load_samples(self, path, guided=False):
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
                    if guided:
                        grads = ast.literal_eval(row[-1])
                        grads = [float(x) for x in grads] 
                    else:
                        grads = []
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, guid, grads))
                except:
                    pass
        return samples


class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path, guided=False):
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
                        samples.append((sentence1, sentence2, label, [], []))
                except:
                    pass
        return samples


class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path, guided=False):
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
                    if guided:
                        grads = ast.literal_eval(row[-1])
                        grads = [float(x) for x in grads] 
                    else:
                        grads = []
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label, guid, grads))
                except:
                    pass
        return samples

class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3

    def load_samples(self, path, guided=False):
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
                        samples.append((sentence1, sentence2, label, [], []))
                except:
                    pass
        return samples


class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path, guided=False):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[5]
                    context = str(row[0])
                    start_ending = str(row[-1])
                    endings = row[1:5]
                    label = int(row[6])
                    if guided:
                        grads = ast.literal_eval(row[-1])
                        grads = [float(x) for x in grads] 
                    else:
                        grads = []
                    samples.append((context, start_ending, endings, label, guid, grads))
                except:
                    pass
        return samples


class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path, guided=False):
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
                    samples.append((context, start_ending, endings, label, [], []))
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

    def __init__(self, path, processor,passed_tokenizer,guided=False):
        self.samples = processor.load_samples(path,guided=guided)
        self.cache = {}
        self.tokenizer = passed_tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]
            if args.task in ('SNLI', 'MNLI', 'QQP', 'MRPC', 'TwitterPPDB'):
                sentence1, sentence2, label, guid, grads = sample
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
                    context, ending_start, endings, label, guid, grads = sample
                    input_ids, segment_ids, attention_mask = encode_mc_inputs(
                        context, ending_start, endings, self.tokenizer
                    )
                else:
                    context, ending_start, endings, label, guid, grads = sample
                    input_ids, attention_mask = encode_mc_inputs(
                        context, ending_start, endings, self.tokenizer
                    ) 
            label_id = encode_label(label)
            if grads is not None:
                grads = cuda(torch.tensor(grads))
            if args.model == 'bert-base-uncased':
                res = ((input_ids, segment_ids, attention_mask), label_id, guid, grads)
            else:
                res = ((input_ids, attention_mask), label_id, guid, grads)
            self.cache[i] = res
        return res


class Model(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model)
        self.classifier = nn.Linear(768, n_classes)
        if args.task in ("SWAG"):
            self.n_choices = -1

    def forward(self, input_ids, segment_ids, attention_mask):
        # On SWAG and HellaSWAG, collapse the batch size and
        # choice size dimension to process everything at once
        if args.task in ('SWAG', 'HellaSWAG'):
            self.n_choices = input_ids.size(1)
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
            logits = logits.view(-1, self.n_choices)
        else:
            cls_output = transformer_outputs[0][:, 0]
            logits = self.classifier(cls_output)
        return logits    
        

def smoothing_label(target, smoothing):
    """Label smoothing"""
    _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
    confidence = 1. - smoothing
    smoothing_value = smoothing / (_n_classes - 1)
    one_hot = cuda(torch.full((_n_classes,), smoothing_value))
    model_prob = one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), confidence)
    return model_prob

def train(d1,d2=None):
    """Fine-tunes pre-trained model on training set."""

    model.train()
    train_loss = 0.
    if args.aum_guide or args.grad_guide:
        train_loader2 = tqdm(load(d2,args.batch_size,True))
    train_loader1 = tqdm(load(d1, args.batch_size, True))
    optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
    if args.aum_guide:
        for i, (dataset1,dataset2) in enumerate(zip(train_loader1,train_loader2)):
            input1, label1, guid1, grads1 = dataset1
            input2, label2, guid2, grads2 = dataset2
            #grads1 = cuda(torch.tensor(grads1))
            #grads2 = cuda(torch.tensor(grads2))
            optimizer.zero_grad()
            if args.model == 'bert-base-uncased':
                logits1 = model(*input1)
                logits2 = model(*input2)
            else:
                logits1 = model(input1[0],None,input1[1])
                logits2 = model(input2[0],None,input2[1])
            loss1 = criterion(logits1,label1)
            loss2 = criterion(logits2,label2)
            if args.grad_guide:
                logits1.retain_grad()
                logits2.retain_grad()
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            
            if args.ls:
                if args.model == 'bert-base-uncased':
                    if args.task == 'SNLI': smoothing_value = 0.001
                    elif args.task == 'QQP': smoothing_value = 0.03
                    elif args.task == 'SWAG' : smoothing_value = 0.3
                elif args.model == 'roberta-base':
                    if args.task == 'SNLI' : smoothing_value = 0.003
                    elif args.task == 'QQP': smoothing_value = 0.03
                    elif args.task == 'SWAG' : smoothing_value = 0.3
                label1 = smoothing_label(label1,smoothing_value)
                label2 = smoothing_label(label2,smoothing_value)
            else:
                label1 = F.one_hot(label1,num_classes=logits1.shape[1])
                label2 = F.one_hot(label2,num_classes=logits2.shape[1])
                
            if args.grad_guide:
                cosine_similarity = nn.CosineSimilarity(dim=1,eps=1e-6)
                current_grad1 = logits1.grad.data.abs()
                current_grad2 = logits2.grad.data.abs()
                
                similar1 = torch.clone(logits1)
                similar_label1 = torch.clone(label1)
                similar2 = torch.clone(logits2)
                similar_label2 = torch.clone(label2)
                dissimilar1 = torch.clone(logits1)
                dissimilar_label1 = torch.clone(label1)
                dissimilar2 = torch.clone(logits2)
                dissimilar_label2 = torch.clone(label2)

                for j in range(0,logits1.shape[0]):
                    similarity_map = cosine_similarity(grads1[j].unsqueeze(0),current_grad2)
                    max_idx = torch.argmax(similarity_map).item()
                    min_idx = torch.argmin(similarity_map).item()
                    similar1[j] = logits2[max_idx]
                    similar_label1 = label2[max_idx]
                    dissimilar1[j] = logits2[min_idx]
                    dissimilar_label1[j] = label2[min_idx]
                alpha = 0.4
                lam = np.random.beta(alpha,alpha)
                id1_logit = logits1 * lam + similar1*(1-lam)
                id1_label = label1 * lam + similar_label1 * (1-lam)
                ood1_logit = logits1 * lam + dissimilar1*(1-lam)
                ood1_label = label1 * lam + dissimilar_label1*(1-lam)
                id_loss1 = torch.mean(torch.sum(-id1_label * torch.log_softmax(id1_logit, dim=-1), dim=0))
                ood_loss1 = torch.mean(torch.sum(-ood1_label * torch.log_softmax(ood1_logit, dim=-1), dim=0))    

                for j in range(0,logits2.shape[0]):
                    similarity_map = cosine_similarity(grads2[j].unsqueeze(0), current_grad1)
                    max_idx = torch.argmax(similarity_map).item()
                    min_idx = torch.argmin(similarity_map).item()
                    similar2[j] = logits1[max_idx] 
                    similar_label2[j] = label1[max_idx]
                    dissimilar2[j] = logits1[min_idx]
                    dissimilar_label2 = label1[min_idx]
                id2_logit = logits2 * lam + similar2*(1-lam)
                id2_label = label2 * lam + similar_label2 * (1-lam)
                ood2_logit = logits2 * lam + dissimilar2*(1-lam)
                ood2_label = label2 * lam + dissimilar_label2*(1-lam)
                id_loss2 = torch.mean(torch.sum(-id2_label * torch.log_softmax(id2_logit, dim=-1), dim=0))
                ood_loss2= torch.mean(torch.sum(-ood2_label * torch.log_softmax(ood2_logit, dim=-1), dim=0))    

                if args.wo_similar:
                    loss = loss1 * 0.9 + ood_loss1 * 0.1
                    loss_ = loss2 * 0.9 + ood_loss2 * 0.1                
                elif args.wo_dissimilar:
                    loss = loss1 * 0.9 + id_loss1 * 0.1 
                    loss_ = loss2 * 0.9 + id_loss2 * 0.1
                else:
                    loss = loss1 * 0.8 + id_loss1 * 0.1 + ood_loss1 * 0.1
                    loss_ = loss2 * 0.8 + id_loss2 * 0.1 + ood_loss2 * 0.1                
                total_loss = (loss + loss_)/2
                train_loss += total_loss.item()
                total_loss.backward()
                if i==0:
                    pass
                else:
                    train_loader1.set_description(f'train loss = {(train_loss / i):.6f}')
                    train_loader2.set_description(f'train loss = {(train_loss / i):.6f}')
            else:
                alpha = 0.4
                lam = np.random.beta(alpha,alpha)
                mixup_logits = logits1 * lam + logits2 * (1-lam)
                mixup_label = label1 * lam + label2 * (1-lam)
                #mixup_loss = criterion(mixup_logits,mixup_label)
                mixup_loss = torch.mean(torch.sum(-mixup_label * torch.log_softmax(mixup_logits, dim=-1), dim=0))
                loss = loss1 * 0.4 + loss2 * 0.4 + mixup_loss * 0.2
                train_loss += loss.item()
                if i==0:
                    pass
                else:
                    train_loader1.set_description(f'train loss = {(train_loss / i):.6f}')
                    train_loader2.set_description(f'mixup loss = {(mixup_loss.item() / i):.6f}')
            if args.max_grad_norm > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        return train_loss / (len(train_loader1)+len(train_loader2))
    else:
        for i, dataset in enumerate(train_loader1):
            inputs, label, guid, grads = dataset
            optimizer.zero_grad()
            if args.model == 'bert-base-uncased':
                logits = model(*inputs)
            else:
                logits = model(inputs[0],None,inputs[1])
            if args.data_categorize:
                logits.retain_grad()
            if args.ls:
                if args.model == 'bert-base-uncased':
                    if args.task == 'SNLI': smoothing_value = 0.001
                    elif args.task == 'QQP': smoothing_value = 0.03
                    elif args.task == 'SWAG' : smoothing_value = 0.3
                elif args.model == 'roberta-base':
                    if args.task == 'SNLI' : smoothing_value = 0.003
                    elif args.task == 'QQP': smoothing_value = 0.03
                    elif args.task == 'SWAG' : smoothing_value = 0.3
                label = smoothing_label(label,smoothing_value)
                #loss = torch.mean(torch.sum(-label * torch.log_softmax(logits, dim=-1), dim=0))
                loss = F.kl_div(F.log_softmax(logits, 1), label, reduction='sum')
            else:
                loss = criterion(logits,label)
            
            loss.backward()
            if args.data_categorize:
                records = aum_calculator.update(logits, label, guid, logits.grad.data.abs().tolist())
            train_loss += loss.item()
            if i==0: 
                pass
            else:
                train_loader1.set_description(f'train loss = {(train_loss / i):.6f}')
            if args.max_grad_norm > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        return train_loss / len(train_loader1)


def evaluate(dataset):
    """Evaluates pre-trained model on development set."""

    model.eval()
    eval_loss = 0.
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    #for i, (inputs, label, guid) in enumerate(eval_loader, 1):
    for i, d in enumerate(eval_loader):
        inputs, label, _, _ = d
        with torch.no_grad():
            if args.model == 'bert-base-uncased':
                logits = model(*inputs)
            else:
                logits = model(inputs[0],None,inputs[1])
            loss = criterion(logits,label)

        eval_loss += loss.item()
        if i == 0:
            pass
        else:
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


criterion = nn.CrossEntropyLoss()

if args.train_path:
    if args.aum_guide or args.grad_guide:
        train_high = TextDataset(args.train_path.replace(".tsv","_high.tsv"), processor, tokenizer, guided=True)
        train_low = TextDataset(args.train_path.replace(".tsv","_low.tsv"), processor, tokenizer, guided=True)
        print(f'train high samples = {len(train_high)}')
        print(f'train low samples = {len(train_low)}')
    else:
        train_dataset = TextDataset(args.train_path, processor,tokenizer)
        print(f'train samples = {len(train_dataset)}')
if args.dev_path:
    dev_dataset = TextDataset(args.dev_path, processor,tokenizer)
    print(f'dev samples = {len(dev_dataset)}')
if args.test_path:
    test_dataset = TextDataset(args.test_path, processor,tokenizer)
    print(f'test samples = {len(test_dataset)}')


if args.do_train:
    print('*** training ***')
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        if args.aum_guide or args.grad_guide:
            train_loss = train(d1=train_high, d2=train_low)
        else:       
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
    if args.data_categorize:
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

    for i, d in enumerate(test_loader):
        inputs, label, _, _ = d
        with torch.no_grad():
            if args.model == 'bert-base-uncased':
                logits = model(*inputs)
            else:
                logits = model(inputs[0],None,inputs[1])
            loss = criterion(logits,label)

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
