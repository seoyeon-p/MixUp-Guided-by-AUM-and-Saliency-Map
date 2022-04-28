This repository contains codes for the following paper:

Seo Yeon Park, Cornelia Caragea : On the Calibration of Pre-trained Language Models using Mixup Guided by Area Under the Margin and Saliency (ACL 2022)

If you would like to refer to it, please cite the paper mentioned above.



## Datasets
We use datasets released by [https://github.com/shreydesai/calibration](https://drive.google.com/file/d/1ro3Q7019AtGYSG76KeZQSq35lBi7lU3v/view). To download out-of-domain datasets (MNLI/TwitterPPDB/HellaSWAG), visit the aforementioned link. 
For the in-domain datasets (SNLI/QQP/SWAG), we download the datasets from the previous link and preprocess the datasets to let each sample to has a unique id. Full datasets are provided in this [link](https://drive.google.com/drive/folders/1xFxPI71mPgx81464yWbOt2QiFBgDTLVq?usp=sharing). In this link, we also provide datasets that are split by Area Under Margins (AUMs) on BERT. Specifically, you can find out train_high.tsv and train_low.tsv, in which train_high.tsv file contains samples that are easy-to-learn and train_low.tsv file contains samples that are hard-to-learn/ambiguous in terms of BERT model. 

Note that our implementation is based on the implementation provided by [this repository](https://github.com/shreydesai/calibration). 


## Requirements
Configure the environments using the below command. Our experiments are done by using python 3.7 using miniconda3:

```
conda create env -n cb python=3.7
conda activate cb
pip install -r requirements.txt
```


## Data Categorization
In our proposed method, we first measure AUMs of all training samples to categorize datasets into easy-to-learn and hard-to-learn/ambiguous. Below we provide an example script for measuring AUM for QQP on BERT.

```
export DEVICE=0
export MODEL="bert-base-uncased"  # options: bert-base-uncased, roberta-base
export TASK="QQP"  # options: SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG
export MAX_SEQ_LENGTH=512

if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi

if [ $MODEL = "roberta-base" ]; then
    BATCH_SIZE=32
    LEARNING_RATE=1e-5
    WEIGHT_DECAY=0.1
fi


python3 train.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --ckpt_path "ckpt/${TASK}_${MODEL}_data_categorize.pt" \
    --output_path "output/${TASK}_${MODEL}.json" \
    --train_path "calibration_data/${TASK}/train.tsv" \
    --dev_path "calibration_data/${TASK}/dev.tsv" \
    --test_path "calibration_data/${TASK}/test.tsv" \
    --epochs 3 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_train \
    --do_evaluate \
    --data_categorize

```

After you finish measuring AUMs on BERT, the file is generated which contains AUM records of each sample that is leveraged to categorize datasets, in the output folder. We categorize training samples into easy-to-learn and hard-to-learn/ambiguous sets by executing the following scripts. We use the median AUMs over full training samples, which are 3.5/4.4/2.5 for BERT, and 3.4/4.0/2.7 for RoBERTa on SNLI/QQP/SWAG.


```
python3 generate_data_category.py \
    --task QQP \
    --th 4.4 \
    --model bert-base-uncased \
    --aum   
```



## MixUp Using Saliency Signals
Then, we conduct MixUp on easy-to-learn and hard-to-learn/ambiguous samples by leveraging gradient-based saliency signals by using the following scripts. We also support multiple GPUs settings. To do this, please include --multigpu at the end of the scripts.

```
export DEVICE=0
export MODEL="bert-base-uncased"  # options: bert-base-uncased, roberta-base
export TASK="SNLI"  # options: SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG
export MAX_SEQ_LENGTH=512

if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi

if [ $MODEL = "roberta-base" ]; then
    BATCH_SIZE=32
    LEARNING_RATE=1e-5
    WEIGHT_DECAY=0.1
fi


python3 train.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --ckpt_path "ckpt/${TASK}_${MODEL}.pt" \
    --output_path "output/${TASK}_${MODEL}.json" \
    --train_path "calibration_data/${TASK}/train.tsv" \
    --dev_path "calibration_data/${TASK}/dev.tsv" \
    --test_path "calibration_data/${TASK}/test.tsv" \
    --epochs 3 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_train \
    --do_evaluate \
    --aum_guid \
    --grad_guid

```

## Evaluating on in-, out-of-domain test sets

To evaluate the fine-tuned model on an out-of-domain test set, execute the below scripts

```
export DEVICE=0
export MODEL="bert-base-uncased"  
export TASK="MNLI"  # options: SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG
export MAX_SEQ_LENGTH=256

if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi

python3 train.py \
    --device $DEVICE \
    --model $MODEL \
    --task $TASK \
    --ckpt_path "ckpt/SNLI_${MODEL}.pt" \
    --output_path "output/${TASK}_${MODEL}.json" \
    --test_path "calibration_data/${TASK}/test.txt" \
    --batch_size $BATCH_SIZE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --do_evaluate 
```

Then, we evaluate model performance (accuracy) and calibration (ECEs) using the output files dumped in the previous step. 

```
export TEST_PATH="./output/SNLI_bert-base-uncased.json"

python3 calibrate.py \
    --test_path $TEST_PATH \
    --do_evaluate
```
