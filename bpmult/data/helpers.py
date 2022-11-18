#!/usr/bin/env python3

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from transformers import DistilBertTokenizer #, BertTokenizer, BertModel
from torch.utils.data import DataLoader

from mmbt.data.dataset import JsonlDataset
from mmbt.data.vocab import Vocab


def get_transforms(args):
    
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path, continuos=False):
    if continuos:
        label_freqs = Counter()
        #data_labels_g = [i/100 for i in range(101)]
        data_labels_g = [2.25, 2.5, -1.33333333333]
        label_freqs.update(data_labels_g)
        data_labels = [json.loads(line)["label"] for line in open(path)]
    else:
        label_freqs = Counter()
        data_labels = [json.loads(line)["label"] for line in open(path)]

    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["mmtrvat", "mmtrvapt"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):
    
    bsz = len(batch)
    
    text_tensor = segment_tensor = mask_tensor = None
    lens = [len(row[0]) for row in batch]
    max_seq_len = max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()

    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    if args.model in ["mmtrvat",  "mmtrvapt"]:
        img_tensor = torch.stack([row[2] for row in batch])
        
    genres = None
    poster = None
    audio = None
    if args.task in ["moviescope", "mmimdb"]:
        if args.model in ["mmtrvapt"]:
            img_lens = [row[4].shape[1] for row in batch]
            img_min_len = min(img_lens)
            audio = torch.stack([row[4][..., :img_min_len] for row in batch])
        if args.visual in ["poster", "both"]:
            poster = torch.stack([row[5] for row in batch])
            
    elif args.task in ["cmu-mosei", "cmu-mosi", "counseling", "iemocap"]:
        if args.model in ["mmtrvat"]:
            img_lens = [row[4].shape[1] for row in batch]
            img_min_len = min(img_lens)
            audio = torch.stack([row[4][..., :img_min_len] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        if args.task == "cmu-mosi":
            tgt_tensor = torch.cat([row[3] for row in batch])
        else:
            tgt_tensor = torch.cat([row[3] for row in batch]).long()
    
    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        
        mask_tensor[i_batch, :length] = 1
    
    if args.task in ["moviescope", "mmimdb", "cmu-mosei", "cmu-mosi", "counseling", "iemocap"]:
        if args.model in ["mmtrvapt"]:
            return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, audio, poster
        elif args.model in ["mmtrvat"]:
            return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, audio
        else:
            return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, poster
    else:
        return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, genres


def get_data_loaders(args, data_all=None, partition_index=None):
    
    if args.model in ["mmtrvapt", "mmtrvat"]:
        tokenizer = (
            BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        )
    else:
        tokenizer = (str.split)

    transforms = get_transforms(args)

    if args.task != "cmu-mosi":
        args.labels, args.label_freqs = get_labels_and_frequencies(
            os.path.join(args.data_path, args.task, "train.jsonl")
        )
    else:
        args.labels, args.label_freqs = get_labels_and_frequencies(
            os.path.join(args.data_path, args.task, "train.jsonl"), continuos=True
        )
    
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)
    
    if args.train_type == "split":

        train = JsonlDataset(
            os.path.join(args.data_path, args.task, "train.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        args.train_data_len = len(train)

        dev = JsonlDataset(
            os.path.join(args.data_path, args.task, "dev.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        collate = functools.partial(collate_fn, args=args)

        train_loader = DataLoader(
            train,
            batch_size=args.batch_sz,
            shuffle=True,
            num_workers=args.n_workers,
            collate_fn=collate,
            drop_last=True,
        )

        val_loader = DataLoader(
            dev,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        test_set = JsonlDataset(
            os.path.join(args.data_path, args.task, "test.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        return train_loader, val_loader, test_loader
    
    else:
        dev_size = int(len(data_all)*0.2)
        train_size = len(data_all)-dev_size
        k = partition_index
        dev_start = k*dev_size
        dev_end = (k+1)*dev_size
        
        if k == 0:
            train_data = data_all[dev_end:]
        elif k == 9:
            train_data = data_all[:dev_start]
        else:
            train_data = data_all[:dev_start] + data_all[dev_end:]
        dev_data = data_all[dev_start:dev_end]
        
        test_size = int(len(train_data)*0.1)
        
        train = JsonlDataset(
            os.path.join(args.data_path, args.task, "train.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
            train_data[test_size:],
        )

        args.train_data_len = len(train)

        dev = JsonlDataset(
            os.path.join(args.data_path, args.task, "dev.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
            dev_data,
        )

        collate = functools.partial(collate_fn, args=args)

        train_loader = DataLoader(
            train,
            batch_size=args.batch_sz,
            shuffle=True,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        val_loader = DataLoader(
            dev,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )
        
        test_set = JsonlDataset(
            os.path.join(args.data_path, args.task, "test.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
            train_data[:test_size],
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )
        
        return train_loader, test_loader, val_loader
