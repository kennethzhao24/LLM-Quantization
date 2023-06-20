import random

from datasets import load_dataset
from transformers import AutoTokenizer


def get_sample_data(dataset, model, nsamples, seq_len):
    if dataset == 'wikitext2':
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    if dataset == 'ptb':
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    # sample data
    random.seed(42)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_dataloader(dataset, model):
    if dataset == 'wikitext2':
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        testloader = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if dataset == 'ptb':
        valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        testloader = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')
    return testloader