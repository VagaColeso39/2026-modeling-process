import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import *

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir=DATA_DIR, transform=None, max_images=MAX_IMAGES,image_size=IMAGE_SIZE):
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size

        self.image_files = []
        if os.path.exists(image_dir):
            self.image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSION)])

        if max_images and len(self.image_files) > max_images:
            random.seed(42)
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:max_images]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(e, "Ошибка при работе с изображением")
            return torch.zeros(1, self.image_size, self.image_size)

def get_transforms(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        AddGaussianNoise(mean=0.0, std=0.01)
    ])

def create_dataloader(image_dir=DATA_DIR, batch_size=BATCH_SIZE,
                      max_images=MAX_IMAGES, image_size=IMAGE_SIZE,
                      num_workers=NUM_WORKERS, shuffle=True):
    transform = get_transforms(image_size)
    dataset = ChestXrayDataset(
        image_dir=image_dir,
        transform=transform,
        max_images=max_images,
        image_size=image_size
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Нет изображений в {image_dir}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )