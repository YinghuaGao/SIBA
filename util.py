import numpy as np
import torch
import os
import random

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def generate_poisoned_trainset(dataset, uap, mask, target, class_order):
    dataset_ = list()
    mask = mask.repeat([3, 1, 1])
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        if i in class_order:
            img = img + uap * mask
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, target))
        else:
            dataset_.append((img, data[1]))         
    return dataset_

def generate_poisoned_testset(dataset, trigger, target, mask):
    dataset_ = list()
    mask = mask.repeat([3, 1, 1])
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        img = img + trigger*mask
        img = torch.clamp(img, 0, 1)          
        dataset_.append((img, target))
    return dataset_

def set_random_seed(seed = 10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MyDataset(torch.utils.data.Dataset):
   
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx][0], self.data[idx][1]
        if self.transform:
            sample = self.transform(sample)
        return (sample, label)
    

def train_step(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test_step(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc