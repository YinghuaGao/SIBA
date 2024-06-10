import math
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import models
from util import *

parser = argparse.ArgumentParser(description='Optimize SIBA trigger with pretrained surrogate model')
parser.add_argument('--surrogate_model', type=str, default='resnet18', 
                    choices=['resnet18', 'resnet34', 'vgg16_bn', 'vgg19_bn'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--y_target', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--k', type=int, default=100)
parser.add_argument('--epsilon', type=float, default=8.0)
parser.add_argument('--step_decay', type=float, default=0.8)
parser.add_argument('--epoch_step', type=int, default=5)
parser.add_argument('--save_surrogate', type=str, default='save_surrogate')
parser.add_argument('--save_trigger', type=str, default='save_trigger')
args = parser.parse_args()


def generate_trigger(model, 
                     loader, 
                     epochs, 
                     eps, 
                     epoch_step=5, 
                     k=5,  
                     beta = 12, 
                     step_decay = 0.8, 
                     y_target = None, 
                     loss_fn = None):
    
    _, (x_val, y_val) = next(enumerate(loader))
    batch_delta = torch.zeros_like(x_val)
    delta = batch_delta[0]
    
    if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    beta = torch.cuda.FloatTensor([beta])
    def clamped_loss(output, target):
        loss = torch.mean(torch.min(loss_fn(output, target), beta))
        return loss

    batch_delta.requires_grad_()
    loss_total = []
    mask_total = []
    delta_total = []
    for epoch in range(epochs):
        eps_step = eps * step_decay
        losses = 0
        for i, (x_val, y_val) in enumerate(loader):
            batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])
            y_val = torch.ones(size = y_val.shape, dtype = y_val.dtype) * y_target
            perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
            outputs = model(perturbed)
            pred_loss = clamped_loss(outputs, y_val.cuda())
            loss = pred_loss
            losses += loss.item()
            loss.backward()
            
            if i == 0 and epoch%epoch_step == 0:
                z = batch_delta.grad.data.mean(dim = 0).abs().sum(dim=0).reshape(-1)
                _, idx = torch.sort(z, descending=True)
                perturb_idx = idx[:k]
                mask = torch.zeros([x_val.shape[2], x_val.shape[3]])
                for j in perturb_idx:
                    mask[math.floor(j/x_val.shape[2])][j%x_val.shape[3]] = 1 
            
            grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
            delta = delta - grad_sign * eps_step
            delta = delta * mask.repeat(3, 1, 1)
            delta = torch.clamp(delta, -eps, eps)
            batch_delta.grad.data.zero_()
        
        loss_total.append(losses/(i+1))
        mask_total.append(mask)
        delta_total.append(delta)
        print('Epoch {} / {}'.format(epoch+1, epochs), 'loss: {:3f}'.format((losses/(i+1))) )
         
    min_index = loss_total.index(min(loss_total))
    print('optimal loss: ', loss_total[min_index])
    return delta_total[min_index].data, mask_total[min_index]


def main(args):
    set_random_seed(args.seed)

    train_dataset = datasets.CIFAR10(root='data', 
                                    train=True, 
                                    transform=transforms.ToTensor(), 
                                    download=True)
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=4) 

    epochs = args.epochs
    eps = args.epsilon / 255
    beta = 10

    model_pretrain = getattr(models, args.surrogate_model)(num_classes=10).to(device)
    model_pretrain.load_state_dict(torch.load(os.path.join(args.save_surrogate, 'benign_model.th')))
    model_pretrain = model_pretrain.cuda()
    model_pretrain.eval()

    save_dir = args.save_trigger
    os.makedirs(save_dir, exist_ok=True)
    uap, mask = generate_trigger(model_pretrain, 
                                train_loader, 
                                epochs, 
                                eps, 
                                args.epoch_step,
                                args.k, 
                                beta, 
                                step_decay=args.step_decay, 
                                y_target=args.y_target)

    np.save('{}/uap.npy'.format(save_dir), uap.detach().cpu().numpy())
    np.save('{}/mask.npy'.format(save_dir), mask.detach().cpu().numpy())

if __name__ == '__main__':
    main(args)