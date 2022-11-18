#!/usr/bin/env python3

import json
import numpy as np
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import pickle
import h5py
import torch
from torch.utils.data import Dataset

from mmbt.utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms_, vocab, args, data_dict=None):
        if data_dict is not None:
            self.data = data_dict
        else:
            self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        self.max_seq_len = args.max_seq_len
        
        self.transforms = transforms_
        self.vilbert_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406, 0.456, 0.485],
                    std=[1., 1., 1.],
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = segment = None
        if self.args.task in ["moviescope", "cmu-mosei", "cmu-mosi", "mmimdb", "counseling", "iemocap"]:
            sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["synopsis"])[:(self.max_seq_len - 1)]
            )
            segment = torch.zeros(len(sentence))
        
        else:
            sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["text"])[:(self.args.max_seq_len - 1)]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            if type(self.data[index]["label"]) != list:
                label[
                    [self.args.labels.index(self.data[index]["label"])]
                ] = 1
            else:
                label[
                    [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
                ] = 1
        else:
            if self.args.task == "cmu-mosi":
                label = torch.FloatTensor(
                    [self.data[index]["label"]]
                )
            else:
                label = torch.LongTensor(
                    [self.args.labels.index(self.data[index]["label"])]
                )

        image = None

        if self.args.model in ["mmtrvat", "mmtrvapt"]:
            if self.args.task == "moviescope":
                if self.args.visual in ["video", "both"]:
                    file = open(os.path.join(self.data_dir, '200F_VGG16', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    image = torch.from_numpy(data).squeeze(0)
                
                poster = None
                if self.args.visual in ["poster", "both"]:
                    file = open(os.path.join(self.data_dir, 'PosterFeatures', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    poster = torch.from_numpy(data).squeeze(0)

            elif self.args.task == "mmimdb":
                if self.args.visual in ["video", "both"]:
                    file = open(os.path.join(self.data_dir, 'glove', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file)['glove']
                    image = torch.from_numpy(data).squeeze(0)

                poster = None
                if self.args.visual in ["poster", "both"]:
                    with h5py.File(os.path.join(self.data_dir, f'multimodal_imdb.hdf5'), 'r') as file:
                        #data = np.array(f['d1'], dtype=np.float32)
                        ind = pickle.load(open(os.path.join(self.data_dir,"indices.pkl"), "rb"))
                        data = file['vgg_features'][ind[self.data[index]["id"]]]
                        poster = torch.from_numpy(data).unsqueeze(0)
 
            elif self.args.task == "counseling":
                if self.args.visual in ["video", "both"]:
                    file = open(os.path.join(self.data_dir, 'glove', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file)['glove']
                    image = torch.from_numpy(data).squeeze(0)
            elif self.args.task == "cmu-mosei":
                if self.args.visual in ["video", "both"]:
                    with open(os.path.join(self.data_dir, 'Vision', f'{self.data[index]["task"]}', f'{str(self.data[index]["id"])}.p'), 'rb') as file:
                
                        image = torch.load(file).float()
                        
            elif self.args.task == "iemocap":
                if self.args.visual in ["video", "both"]:
                    with open(os.path.join(self.data_dir, 'Vision', f'{self.data[index]["task"]}', f'{str(self.data[index]["id"])}.p'), 'rb') as file:
                
                        image = torch.load(file).float()
                        
            elif self.args.task == "cmu-mosi":
                if self.args.visual in ["video", "both"]:
                    ID = self.data[index]["id"]
                    PART = self.data[index]["part"]
                    with open(os.path.join(self.data_dir, 'mosi_data.pkl'), 'rb') as f:
                    #with open(os.path.join(self.data_dir, 'mosi_data_noalign.pkl'), 'rb') as f:
                        data = pickle.load(f)
                        data = np.array(data[PART]['vision'][ID], dtype='float32')
                        image = torch.from_numpy(data).squeeze(0)

                
        audio = None
        if self.args.model in ["mmtrvat",  "mmtrvapt"]:
            if self.args.task == "moviescope":
                if self.args.orig_d_a == 96:
                    file = open(os.path.join(self.data_dir, 'Melspectrogram', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    audio = torch.from_numpy(data).type(torch.FloatTensor)
                else:
                    file = open(os.path.join(self.data_dir, 'MelgramPorcessed', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    data = torch.from_numpy(data).type(torch.FloatTensor).squeeze(0)
                    audio = torch.cat([frame for frame in data[:4]], dim=1)
            elif self.args.task == "cmu-mosei":
#                with open(os.path.join(self.data_dir,'mosei_senti_data_noalign.pkl'), 'rb') as f:
 #                   data = pickle.load(f)
              #  data_audio = data[self.data[index]['task']]['audio'][self.data[index]['id']]
               # audio = torch.from_numpy(np.array(data_audio, dtype=np.float32)).squeeze(0)
                with open(os.path.join(self.data_dir, 'Audio', f'{self.data[index]["task"]}', f'{str(self.data[index]["id"])}.p'), 'rb') as file:
                    audio = torch.load(file).float()
                    
            elif self.args.task == "iemocap":
                with open(os.path.join(self.data_dir, 'Audio', f'{self.data[index]["task"]}', f'{str(self.data[index]["id"])}.p'), 'rb') as file:
                    audio = torch.load(file).float()

            elif self.args.task == "cmu-mosi":
                ID = self.data[index]["id"]
                PART = self.data[index]["part"]
                with open(os.path.join(self.data_dir, 'mosi_data.pkl'), 'rb') as f:
                #with open(os.path.join(self.data_dir, 'mosi_data_noalign.pkl'), 'rb') as f:
                    data = pickle.load(f)
                    data = np.array(data[PART]['audio'][ID], dtype='float32')
                    audio = torch.from_numpy(data).squeeze(0)

            elif self.args.task == "mmimdb":
                file = open(os.path.join(self.data_dir, 'BoW', f'{str(self.data[index]["id"])}.p'), 'rb')
                data = pickle.load(file)['bow']
                audio = torch.from_numpy(data).float().unsqueeze(1)
  #              print("BoW as audio loaded", audio.size())
  
            elif self.args.task == "counseling":
                file = open(os.path.join(self.data_dir, 'fasttext', f'{str(self.data[index]["id"])}.p'), 'rb')
                data = pickle.load(file)['fastText']
                audio = torch.from_numpy(data).float().squeeze(0)

        
        if self.args.task in ["moviescope", "mmimdb"]:
            if self.args.model in ["mmtrvapt"]:
                return sentence, segment, image, label, audio, poster
            else:
                return sentence, segment, image, label, poster
                
        elif self.args.task in ["cmu-mosei", "cmu-mosi", "counseling", "iemocap"]:
            if self.args.model in ["mmtrvat"]:
                return sentence, segment, image, label, audio
        else:
            return sentence, segment, image, label
