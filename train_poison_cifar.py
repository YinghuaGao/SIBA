from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import models
import logging
import time
from util import *

parser = argparse.ArgumentParser(description='Train Backdoored Model')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--y_target', type=int, default=0)
parser.add_argument('--poison_rate', type=float, default=0.01)
parser.add_argument('--save_dir', type=str, default='save_backdoor')
parser.add_argument('--save_trigger', type=str, default='save_trigger')
args = parser.parse_args()


def main(args):

    set_random_seed(args.seed)

    train_dataset = datasets.CIFAR10(root='data', 
                                    train=True, 
                                    transform=transforms.ToTensor(), 
                                    download=True)
    test_dataset = datasets.CIFAR10(root='data', 
                                    train=False, 
                                    transform=transforms.ToTensor(), 
                                    download=True)

    uap = np.load('{}/uap.npy'.format(args.save_trigger))
    mask = np.load('{}/mask.npy'.format(args.save_trigger))
    uap = torch.from_numpy(uap)
    mask = torch.from_numpy(mask)
    mask = mask.detach().cpu()
    uap = uap.detach().cpu()

    shuffle = np.random.permutation(len(train_dataset))
    total_poison = int(len(train_dataset)*args.poison_rate)
    k = 0
    class_order = [] 
    for i in shuffle:
        if train_dataset[i][1] != args.y_target and k < total_poison:
            class_order.append(i)
            k += 1

    poison_train_set = generate_poisoned_trainset(train_dataset, uap, mask, args.y_target, class_order)
    poison_test_set = generate_poisoned_testset(test_dataset, uap, args.y_target, mask)

    train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor()])
    poison_train_set = MyDataset(poison_train_set, train_transform)
    poison_train_loader = DataLoader(poison_train_set, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=4)
    clean_test_loader = DataLoader(test_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4)
    trigger_loader = DataLoader(poison_test_set, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4)

    model = getattr(models, args.model)(num_classes=10).to(device)
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4)
    scheduler = MultiStepLR(
            model_optimizer, 
            milestones=[60, 90], 
            gamma=0.1)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = logging.getLogger()
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(args.save_dir, 'output.log')),
                logging.StreamHandler()
            ])
    logger.info(args)
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for epoch in range(args.epochs):
        start = time.time()
        lr = model_optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step(model, criterion, model_optimizer, poison_train_loader)
        cl_test_loss, cl_test_acc = test_step(model, criterion, clean_test_loader)
        po_test_loss, po_test_acc = test_step(model, criterion, trigger_loader)
        scheduler.step()
        end = time.time()
        logger.info(
                '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "backdoor_model.th"))


if __name__ == '__main__':
    main(args)