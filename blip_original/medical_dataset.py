import os
import json

from torch.utils.data import Dataset

from PIL import Image

from .utils import pre_caption
import torch
import numpy as np

node = [
    'normal', 'other finding', 'heart', 'cardiomegaly', 'spine', 'scoliosis', 'pleural', 'effusion', 'thickening',
    'pneumothorax',
    'bone', 'bone fractures', 'lung', 'emphysema', 'pneumonia', 'edema', 'atelectasis', 'clcatrix', 'opacity', 'lesion',
    'mediastinum', 'hernia', 'calcinosis', 'foreign object', 'airspace', 'airspace disease', 'hypoinflation'
]
nodes = ' '.join(node)


# nodes = node
class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=90, prompt='', dataset='', args=None):

        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        # self.ann = self.annotation['train']
        self.ann = self.annotation
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.dataset = dataset
        self.args = args

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = ann['image_path']
        if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':
            image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image = self.transform(image)

        caption = self.prompt + pre_caption(ann['report'], self.max_words)

        knowledge_skg = nodes
        knowledge_tc = ''
        triplet_len = len(ann['triplet'])
        if triplet_len > 30:
            for i in range(30):
                knowledge_tc += ann['triplet'][i]
                if i < 29:
                    knowledge_tc += '-'
        else:
            tri_idx = 0
            for triplet in ann['triplet']:
                knowledge_tc += triplet
                tri_idx += 1
                if tri_idx < triplet_len:
                    knowledge_tc += '-'
        knowledge_tc = pre_caption(knowledge_tc, self.max_words)  # triplet can see each other

        label = np.array(ann['label_index'], dtype=np.float32)

        return image, caption, knowledge_skg, knowledge_tc, label


class generation_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, dataset, args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.ann = self.annotation[split]
        self.transform = transform
        self.image_root = image_root
        self.dataset = dataset
        self.args = args

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        image_path = ann['image_path']
        if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':
            image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image = self.transform(image)

        caption = pre_caption(ann['report'], 90)
        if self.args.SKG_know:
            knowledge = nodes
        else:
            knowledge = ''

        return image, caption, knowledge, image_path


class retrieval_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=90, dataset='', args=None):

        # self.annotation = json.load(open(ann_root+'train_know.json', 'r'))
        self.annotation = json.load(open(ann_root, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.dataset = dataset
        self.args = args

        self.image_id = -1
        self.img_ids = {}

        n = 0
        img_id = 0
        for ann in self.annotation:
            # img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            img_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]
        image_path = ann['image_path']
        if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':
            image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image = self.transform(image)

        caption = pre_caption(ann['report'], self.max_words)

        knowledge_skg = nodes
        knowledge_tc = ''
        triplet_len = len(ann['triplet'])
        if triplet_len > 30:
            for i in range(30):
                knowledge_tc += ann['triplet'][i]
                if i < 29:
                    knowledge_tc += '-'
        else:
            tri_idx = 0
            for triplet in ann['triplet']:
                knowledge_tc += triplet
                tri_idx += 1
                if tri_idx < triplet_len:
                    knowledge_tc += '-'
        knowledge_tc = pre_caption(knowledge_tc, self.max_words)  # triplet can see each other

        self.image_id += 1

        return image, caption, knowledge_skg, knowledge_tc, self.img_ids[self.image_id]


class retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=90, args=None):

        # self.annotation = json.load(open(ann_root + split + '.json', 'r'))
        self.ann = json.load(open(os.path.join(ann_root), 'r'))
        self.annotation = self.ann[split]
        self.transform = transform
        self.image_root = image_root
        self.args = args
        self.dataset = self.args.dataset_name

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image_path'])
            self.img2txt[img_id] = []
            cur_report = []
            cur_report.append(ann['report'])
            for i, caption in enumerate(cur_report):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = ann['image_path']
        if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':
            image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image = self.transform(image)
        if self.args.SKG_know:
            knowledge = nodes
        else:
            knowledge = ''
        return image, index, knowledge


class diagnose_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=90, dataset='', args=None):

        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.ann = self.annotation[0]['train']
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.dataset = dataset
        self.args = args

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = ann['image_path']
        if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':
            image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image = self.transform(image)

        elif self.dataset == 'chexpert':
            image = Image.open(os.path.join(self.image_root, image_path)).convert('RGB')
            image = self.transform(image)

        label = np.array(ann['label_index'])
        caption = ''
        if self.args.SKG_know:
            knowledge = nodes
        else:
            knowledge = ''
            if self.dataset == 'iu_xray' or self.dataset == 'mimic_cxr':
                caption = pre_caption(ann['report'], self.max_words)
                if len(ann['triplet']) > 30:
                    for i in range(30):
                        knowledge += ann['triplet'][i]
                        knowledge += ' '
                else:
                    for triplet in ann['triplet']:
                        knowledge += triplet
                        knowledge += ' '
                knowledge = pre_caption(knowledge, self.max_words)  # triplet can see each other

        return image, caption, knowledge, label


class diagnose_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, dataset, args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.ann = self.annotation[0][split]
        self.transform = transform
        self.image_root = image_root
        self.dataset = dataset
        self.args = args

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        image_path = ann['image_path']
        if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':
            image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image = self.transform(image)

        elif self.dataset == 'chexpert':
            image = Image.open(os.path.join(self.image_root, image_path)).convert('RGB')
            image = self.transform(image)

        caption = ''
        if self.dataset == 'mimic_cxr' or self.dataset == 'iu_xray':
            caption = pre_caption(ann['report'], 90)

        label = np.array(ann['label_index'])
        if self.args.SKG_know:
            knowledge = nodes
        else:
            knowledge = ''

        return image, caption, label, knowledge


class diagnose_knowledge(Dataset):
    def __init__(self, ann, dataset='', args=None):
        self.ann = json.load(open(os.path.join(ann), 'r'))
        self.dataset = dataset
        self.args = args

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        if self.args.SKG_know:
            knowledge = nodes
        else:
            knowledge = self.ann[index]

        return knowledge