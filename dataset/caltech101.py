import os 
import json 
import pickle 
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

TRAIN_IMG_PER_CLASS=30

class caltech101(Dataset):

    classes=[
        'background',
        'off-center face',
        'centered face',
        'leopard',
        'motorbike',
        'accordion',
        'airplane',
        'anchor',
        'ant',
        'barrel',
        'bass',
        'beaver',
        'binocular',
        'bonsai',
        'brain',
        'brontosaurus',
        'buddha',
        'butterfly',
        'camera',
        'cannon',
        'side of a car',
        'ceiling fan',
        'cellphone',
        'chair',
        'chandelier',
        'body of a cougar cat',
        'face of a cougar cat',
        'crab',
        'crayfish',
        'crocodile',
        'head of a crocodile',
        'cup',
        'dalmatian',
        'dollar bill',
        'dolphin',
        'dragonfly',
        'electric guitar',
        'elephant',
        'emu',
        'euphonium',
        'ewer',
        'ferry',
        'flamingo',
        'head of a flamingo',
        'garfield',
        'gerenuk',
        'gramophone',
        'grand piano',
        'hawksbill',
        'headphone',
        'hedgehog',
        'helicopter',
        'ibis',
        'inline skate',
        'joshua tree',
        'kangaroo',
        'ketch',
        'lamp',
        'laptop',
        'llama',
        'lobster',
        'lotus',
        'mandolin',
        'mayfly',
        'menorah',
        'metronome',
        'minaret',
        'nautilus',
        'octopus',
        'okapi',
        'pagoda',
        'panda',
        'pigeon',
        'pizza',
        'platypus',
        'pyramid',
        'revolver',
        'rhino',
        'rooster',
        'saxophone',
        'schooner',
        'scissors',
        'scorpion',
        'sea horse',
        'snoopy (cartoon beagle)',
        'soccer ball',
        'stapler',
        'starfish',
        'stegosaurus',
        'stop sign',
        'strawberry',
        'sunflower',
        'tick',
        'trilobite',
        'umbrella',
        'watch',
        'water lilly',
        'wheelchair',
        'wild cat',
        'windsor chair',
        'wrench',
        'yin and yang symbol',
        ]

    templates = [
        'a photo of a{}.',
        'a painting of a {}.',
        'a plastic {}.',
        'a sculpture of a{}.',
        'a sketch of a {}.',
        'a tattoo of a{}.',
        'a toy{}.',
        'a rendition of a {}.',
        'a embroidered {}.',
        'a cartoon {}.',
        'a {} in a video game.',
        'a plushie {}.',
        'a origami {}.',
        'art of a{}.',
        'graffiti of a{}.',
        'a drawing of a {}.',
        'a doodle of a {}.',
        'a photo of the {}.',
        'a painting of the {}.',
        'the plastic {}.',
        'a sculpture of the {}.',
        'a sketch of the {}.',
        'a tattoo of the {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'the embroidered {}.',
        'the cartoon {}.',
        'the {} in a video game.',
        'the plushie {}.',
        'the origami {}.',
        'art of the {}.',
        'graffiti of the {}.',
        'a drawing of the {}.',
        'a doodle of the {}.',
    ]


    def __init__(self, root, transform=None, train=True):
        self.root = os.path.join(root,'101_ObjectCategories')
        self.train= train
        self.transform = transform

        split = 'train' if self.train else 'test'
        split_file_path = os.path.join(root,'split.json')

        if not os.path.isfile(split_file_path):
            ori_classes = sorted(os.listdir(self.root))
            train_data =[]
            train_labels =[]
            test_data=[]
            test_labels =[]
            for (i, c) in enumerate(ori_classes):
                class_imgs = sorted(os.listdir(os.path.join(self.root, c)))
                n = len(class_imgs)
                random.shuffle(class_imgs)
                train_data.extend([os.path.join(c, im) for im in class_imgs[:TRAIN_IMG_PER_CLASS]])
                train_labels.extend([i for _ in range(TRAIN_IMG_PER_CLASS)])
                test_data.extend([os.path.join(c, im) for im in class_imgs[TRAIN_IMG_PER_CLASS:]])
                test_labels.extend([i for _ in range(n-TRAIN_IMG_PER_CLASS)])

            split_file={
                'train_data':train_data,
                'train_labels': train_labels,
                'test_data':test_data,
                'test_labels':test_labels
            }
            with open(split_file_path, 'w') as f:
                json.dump(split_file,f)

        with open(split_file_path,'r') as f:
            split_data= json.load(f)

        self.data = split_data['{}_data'.format(split)]
        self.labels = split_data['{}_labels'.format(split)]


    def __getitem__(self, index):
        img = self.data[index]
        img = os.path.join(self.root,img)
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def prompts(self, mode='single'):
        if mode == 'single':
            prompts = [[self.templates[0].format(label)] for label in self.classes]
            return prompts
        elif mode == 'ensemble':
            prompts = [[template.format(label) for template in self. templates] for label in self.classes]
            return prompts

    def get_labels(self):
        return np.array(self.labels)

    def get_classes(self):
        return self.classes