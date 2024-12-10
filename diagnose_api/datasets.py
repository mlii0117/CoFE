import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import json
import numpy as np
from blip_original.utils import pre_caption



class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        if args.dataset_name == 'iu_xray':
            self.ann_path = args.ann_path.split('&')[0]     #/mnt/data/chenzicong2/mimic_dia_index.json
            self.image_dir = args.image_dir.split('&')[0]
            self.knowledge_path = '/data/linhaokun/project/dataset/iu_xray/annotation_kg.json'
        elif args.dataset_name == 'mimic_cxr':
            self.ann_path = args.ann_path.split('&')[1]
            self.image_dir = args.image_dir.split('&')[1]
            self.knowledge_path = '/data/linhaokun/project/dataset/MIMIC-CXR/mimic_cxr/annotation_kg.json'
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        with open(self.ann_path, encoding='utf-8') as f:
            self.ann = json.load(f)
            f.close()
        with open(self.knowledge_path, encoding='utf-8') as f:
            self.ann_knowledge = json.load(f)
            f.close()
        self.examples = self.ann[0][self.split]
        self.examples_knowledge = self.ann_knowledge
        self.max_words = 90
        ## create a dict to store the mask
        # self.masks = []
        # self.reports = []

        # for i in range(len(self.examples)):
        #     caption = tokenizer(self.examples[i]['report'])[:self.max_seq_length]   #原来的tokenizer
        #     # caption = tokenizer(self.examples[i]['report'], padding='longest', max_length=250, return_tensors="pt")
        #     # caption = caption['input_ids'][:self.max_seq_length]
        #     self.examples[i]['ids'] = caption
        #     self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
        #     # self.masks.append([1] * len(self.reports[i]))

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        # example = self.examples.loc[idx]
        example = self.examples[idx]
        example_knowledge = self.examples_knowledge[idx]
        # image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        label = np.array(example['label_index'])
        report = pre_caption(example['report'], self.max_words)
        # report_ids = example['ids']     #ids?
        # report_masks = example['mask']      #mask?
        # seq_length = len(report_ids)
        # sample = (image, report_ids, report_masks, seq_length)
        knowledge = ''
        for triplet in example_knowledge['triplet']:
            knowledge += triplet
            knowledge += ' '
        sample = (image, report, knowledge, label)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        example_knowledge = self.examples_knowledge[idx]
        # image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # label = list(example['label_index'])
        # label = torch.FloatTensor(label)
        # label = torch.FloatTensor(np.array(label))
        report = pre_caption(example['report'], self.max_words)
        label = np.array(example['label_index'])
        # report_ids = example['ids']
        # report_masks = example['mask']
        # seq_length = len(report_ids)
        # sample = (image_id, image, report_ids, report_masks, seq_length)
        knowledge = ''
        for triplet in example_knowledge['triplet']:
            knowledge += triplet
            knowledge += ' '
        sample = (image, report, knowledge, label)
        return sample