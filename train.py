import torch
from torch.utils.data import DataLoader
import cv2
import os
import torch
import numpy as np

from model import PortraitNet
from loss import CombinedLoss
from config import IMAGE_DIR, IMAGE_SIZE, BATCH_SIZE, TRAIN_SPLIT, LR, DEVICE, NUM_EPOCHS
from config import scaling_transform, augment_transform
from dataset import PortraitDataset
from utils import evaluate_mean_iou


def runner(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs):
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader):
            images, masks = data
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            del inputs, masks, outputs
            torch.cuda.empty_cache()
            
        mean_iou = evaluate_mean_iou(model, test_dataloader, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader)}), mIOU: {mean_iou}")

        torch.save(model.state_dict(), f'PortraitNet_e{epoch}.pth')

    print("Training Complete!")



if __name__ == "__main__":
    # walk through the image directory
    image_file_paths = []
    masks_file_paths = []
    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            if ".png" in file or ".jpg" in file:
                p = os.path.join(root, file)
                image_file_paths.append(p)
                masks_file_paths.append(p.replace("_image.jpg", "_mask.png"))  # images/1_image.jpg -> images/1_mask.png

    # Preprocess and store the images and masks
    images = []
    masks = []

    for img_path, mask_path in zip(image_file_paths, masks_file_paths):
        # Read the image with alpha channel
        in_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Resize the image and mask
        resized_image = cv2.resize(in_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # Normalize the image
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
        resized_image = resized_image / 255.0

        # Apply scaling transforms
        transformed_image = scaling_transform(resized_image)

        # Store the image and mask
        images.append(transformed_image)
        masks.append(resized_mask)

    # Convert to numpy arrays if you want to apply some transformations later on with torchvision
    images = np.array(images)
    masks = np.array(masks)

    # Create the full dataset
    full_dataset = PortraitDataset(images, masks, transform = augment_transform)    

    # Split images and masks into train and test
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    # Define the model
    model = PortraitNet(num_classes = 2)

    # Define the loss function
    combined_loss = CombinedLoss(alpha = 0.25, gamma = 2.0)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    # Start training
    runner(model, train_dataloader, test_dataloader, optimizer, combined_loss, device = DEVICE, epochs = NUM_EPOCHS)


