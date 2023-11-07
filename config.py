IMAGE_DIR = 'data/images'
IMAGE_SIZE = (224, 224)

TRAIN_SPLIT = 0.8
BATCH_SIZE = 128
LR = 0.0001
NUM_EPOCHS = 100
DEVICE = "cpu"

from torchvision import transforms
import torch
augment_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda x: x + torch.randn(x.size()) * 10),
    transforms.RandomChoice([
        transforms.GaussianBlur(kernel_size=3),
        transforms.GaussianBlur(kernel_size=5)
    ]),
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.4),
    transforms.ColorJitter(brightness=(0.4, 1.7)),
    transforms.ColorJitter(contrast=(0.6, 1.5)),
    transforms.RandomAdjustSharpness(sharpness_factor=(0.8, 1.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
scaling_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees = (-45, 45)),
    transforms.RandomResizedCrop(size = IMAGE_SIZE, scale = (0.5, 1.5)),
    transforms.RandomAffine(degrees = 0, translate = (0.25, 0.25)),
    transforms.ToTensor(),
])