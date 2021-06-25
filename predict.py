import os
from imutils import paths
import imageio

import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import ResNetUNet
from utilities import reverse_transform, reverse_transform_mask
from preprocess import check_dir

WEIGHT_PATH = "./model/pretrained"
USE_BEST_VAL = True
DISPLAY_PLOTS = False
TEST_DIR = "./test/content/img_224/"
SAVE_PATH = "./test/output"
PREFIX = "seg_"

trans = transforms.Compose([
    transforms.ToTensor()
])


class parseTestset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        _, filename = os.path.split(image_path)

        if self.transform:
            image = self.transform(image)  # ToTensor
            image = transforms.Normalize(  # TODO: remove this?
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])(image)
        return image, filename


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on: {device}')

    num_class = 1
    model = ResNetUNet(num_class).to(device)

    if USE_BEST_VAL:
        model.load_state_dict(torch.load(
            os.path.join(WEIGHT_PATH, "best_val_weights.pth")))
    else:
        model.load_state_dict(torch.load(
            os.path.join(WEIGHT_PATH, "latest_weights.pth")))

    test_img_paths = list(paths.list_images(TEST_DIR))
    print(f'found {len(test_img_paths)} images')

    # small batch_size if you are testing on 1 or 2 images
    b_size = min(25, len(test_img_paths))

    test_set = parseTestset(test_img_paths, transform=trans)
    test_loader = DataLoader(test_set, batch_size=b_size,
                             shuffle=True, num_workers=0)

    check_dir(SAVE_PATH)

    model.eval()
    for i, batch_pair in enumerate(tqdm(test_loader)):
        img_batch = batch_pair[0].to(device)
        img_names = batch_pair[1]

        seg_batch = model(img_batch)
        seg_batch = torch.sigmoid(seg_batch)
        for img, seg, filename in zip(img_batch, seg_batch, img_names):
            seg_np = seg.cpu().detach()
            seg_np = reverse_transform_mask(seg_np)
            seg_np = np.where(seg_np > 220, 1, 0)

            img_np = img.cpu()
            img_np = reverse_transform(img_np)
            prod_img = np.multiply(seg_np, img_np).astype("uint8")

            if len(PREFIX) > 0:
                filename = PREFIX + filename

            savename = os.path.join(SAVE_PATH, filename)
            imageio.imwrite(savename, prod_img)
