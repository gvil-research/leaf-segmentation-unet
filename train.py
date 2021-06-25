import os
import glob
import time
import copy
from collections import defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

import albumentations as A

from model import dice_loss, ResNetUNet
from preprocess import check_dir

WEIGHT_PATH = "./model/pretrained"
check_dir(WEIGHT_PATH)

def read_imgs_and_masks(folder_path, display=False):
    """
    Read images and their corresponding mask (optionally show two
    random image-mask pairs)
    """
    mask_paths = glob.glob(folder_path + '/*seg.jpg')
    img_paths = list(map(lambda st: st.replace("seg", "img"), mask_paths))

    if display:
        idx = np.random.randint(low=0, high=len(mask_paths), size=(2,))
        for i in idx:
            mask_img = cv2.imread(mask_paths[i])
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            _, mask_img = cv2.threshold(mask_img, 5, 255, cv2.THRESH_BINARY)

            org_img = cv2.imread(img_paths[i])
            org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

            _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
            ax1.imshow(org_img)

            ax2.imshow(mask_img, cmap='gray')
    return img_paths, mask_paths


class parseDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, augment=None):
        self.img_paths, self.mask_paths = img_paths, mask_paths
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = self.img_paths[idx]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.mask_paths[idx]
        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        if self.augment:
            augmented = self.augment(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            # for images but not for masks
            image = transforms.Normalize([0.485, 0.456, 0.406], [
                                         0.229, 0.224, 0.225])(image)

        return [image, mask]


# use the same transformations for train/val in this example
trans = transforms.Compose([
    transforms.ToTensor()
])

aug = A.Compose([
    A.RandomSizedCrop(min_max_height=(160, 224), height=224, width=224, p=0.5),
    A.PadIfNeeded(min_height=224, min_width=224, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8)
])


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                # save to disk as well
                torch.save(model.state_dict(), os.path.join(WEIGHT_PATH, 'best_val_weights.pth'))
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                scheduler.step()

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s \n'.format(
            time_elapsed // 60, time_elapsed % 60))
        torch.save(model.state_dict(), os.path.join(WEIGHT_PATH, 'latest_weights.pth'))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # folder paths
    TRAIN_PATH = './dataset/train'
    VAL_PATH = './dataset/test'

    train_img_paths, train_img_masks = read_imgs_and_masks(TRAIN_PATH)
    val_img_paths, val_img_masks = read_imgs_and_masks(VAL_PATH)

    train_set = parseDataset(train_img_paths, train_img_masks,
                            transform=trans, augment=aug)
    val_set = parseDataset(val_img_paths, val_img_masks, transform=trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on: {device}')

    num_class = 1
    model = ResNetUNet(num_class).to(device)

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    # start from lr=1e-4 and then slowly decrease
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

    # finally train
    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=30)
