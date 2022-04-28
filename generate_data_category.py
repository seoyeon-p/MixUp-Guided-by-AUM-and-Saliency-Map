import os
import argparse
import csv
import ast
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--th', type=float, default='2.')
parser.add_argument('--aum', action='store_true')
parser.add_argument('--model',type=str)
args = parser.parse_args()

print(args)

path = './output/'+args.task+"_"+args.model+'/aum_values.csv'
original_path = './calibration_data/'+args.task+'/train.tsv'

class SNLIProcessor:
    def load_samples(self, path):
        samples = {}
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            idx = 0
            for row in tqdm(reader, desc=desc):
                if idx == 0: 
                    header = row
                else:
                    guid = row[1]
                    samples[guid] = row
                idx += 1
        return samples, header

class QQPProcessor:
    def load_samples(self, path):
        samples = {}
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            idx = 0
            for row in tqdm(reader, desc=desc):
            #for row in reader:
                if idx == 0:
                    header = row
                else:
                    guid = row[0]
                    samples[guid] = row
                idx += 1
        return samples, header

class SWAGProcessor:
    def load_samples(self, path):
        samples = {}
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            idx = 0
            for row in tqdm(reader, desc=desc):
                if idx == 0: 
                    header = row
                else:
                    guid = row[5]
                    samples[guid] = row
                idx += 1
        return samples, header

def select_processor():
    """Selects data processor using task name."""
    return globals()[f'{args.task}Processor']()

processor = select_processor()
data, header = processor.load_samples(original_path)

with open(path) as f:
    reader = csv.reader(f, delimiter=',')
    desc = f'loading \'{path}\''
    aum_grad_dict = {}
    #for row in tqdm(reader, desc=desc):
    idx = 0
    for row in reader:
        if idx == 0:
            idx += 1
            continue
        guid = row[0]
        aum = row[1]
        magnitude_grads = ast.literal_eval(row[2])
        val_dict = {}
        val_dict['aum'] = aum
        val_dict['grads'] = magnitude_grads
        aum_grad_dict[guid] =val_dict
        idx += 1

high,low = [], []
for d in data:
    try:
        data[d] = "\t".join(data[d]) #data[d].append(aum_grad_dict[d]['grads'])
        data[d] = data[d] + '\t' + str(aum_grad_dict[d]['grads'])
        if args.aum:
            #if float(aum_grad_dict[d]['aum']) < 3.5 :
            if float(aum_grad_dict[d]['aum']) < args.th :
                low.append(data[d])
            else:
                high.append(data[d])
        else:
            if len(low) < int(len(data)/2):
                low.append(data[d])
            else:
                high.append(data[d])
    except:
        pass

if args.aum:
    high_path = './calibration_data/' + args.task + '/train_high.tsv'
    low_path = './calibration_data/' + args.task + '/train_low.tsv'
else:
    high_path = './calibration_data/' + args.task + '/train_random1.tsv'
    low_path = './calibration_data/' + args.task + '/train_random2.tsv'

with open(high_path, 'w', newline='',encoding='utf-8') as output_file:
    output_file.write(str(header)+'\n')
    desc = f'writing \'{high_path}\''
    for item in tqdm(high,desc=desc):
        output_file.write(str(item)+'\n')
          
with open(low_path, 'w', newline='',encoding='utf-8') as output_file:
    output_file.write(str(header)+'\n')
    desc = f'writing \'{low_path}\''
    for item in tqdm(low,desc=desc):
        output_file.write(str(item)+'\n')
