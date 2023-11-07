import argparse
import cv2
import numpy as np
import torch

from model import PortraitNet
from config import IMAGE_SIZE


def segment_image(image):
    # Preprocess the input image
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Forward pass through the U-Net model
    with torch.no_grad():
        output = model(image_tensor)

    # Convert the output to a binary mask (assuming binary segmentation)
    mask = (output > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)

    return mask


def parse_opt():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset of images with text.")

    parser.add_argument("-mp", "--model_path", type=str, required=True,
                    help="Path to the model state-dict.")
    parser.add_argument("-im", "--input_image", type=str, required=True,
                    help="Input image path.")

    return parser.parse_args()


if __name__ == "__main__":
    # args
    args = parse_opt()

    model_path = args.model_path
    input_image_path = args.input_image

    # Define and load the model
    model = PortraitNet(num_classes = 2)
    model.load_state_dict(torch.load(model_path))

    # Read the input image using OpenCV and resize it
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    original_width, original_height = input_image.shape[1], input_image.shape[0]
    input_image_resized = cv2.resize(input_image, IMAGE_SIZE, interpolation = cv2.INTER_AREA)

    segmented_mask = segment_image(input_image_resized)

    # Resize the mask back to the original size
    original_size_mask = cv2.resize(segmented_mask, (original_width, original_height))  # Set original width and height

    # Apply a color map for visualization
    segmented_mask = cv2.applyColorMap(original_size_mask * 255, cv2.COLORMAP_JET)

    # Combine the mask and input image
    combined = np.concatenate([input_image, segmented_mask], axis = 1)

    # Save the output
    cv2.imwrite("output.png", combined)
    

