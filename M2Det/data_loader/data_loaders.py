import os
import cv2
import json
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.transform import Rescale, ToTensor


class BDDDataset(Dataset):
    def __init__(self, image_dir, label_dir, label, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        frames = json.load(open(os.path.join(label_dir, label), 'r'))

        labels = ["bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign", "train", "truck"]

        self.images = list()
        self.boxes = list()
        for frame in frames:
            box = list()
            for label in frame['labels']:
                if 'box2d' not in label:
                    continue
                xy = label['box2d']
                if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                    continue
                box.append([xy['x1'], xy['y1'], xy['x2'], xy['y2'], labels.index(label['category'])])
            self.images.append(frame['name'])
            self.boxes.append(box)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)

        sample = {'data': image, 'bbox': self.boxes[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def collate_fn(self, batch):
        images = list()
        boxes = list()

        for sample in batch:
            images.append(sample['data'])
            boxes.append(sample['bbox'])

        images = torch.stack(images, dim=0)

        return {'data': images, 'bbox': boxes}


class BDDDataLoader(BaseDataLoader):
    def __init__(self, image_dir, label_dir, label, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = transforms.Compose([
            Rescale((320, 320)),
            ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = BDDDataset(image_dir, label_dir, label, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
