# MobileNetV2 UNet for Image Segmentation
This repository contains a TensorFlow/Keras implementation of a U-Net architecture with a MobileNetV2 backbone for the purpose of image segmentation. This hybrid neural network combines the efficiency of MobileNetV2 with the effective architecture of U-Net for semantic segmentation tasks.

MobileNetV2 is used as the encoder, which is pre-trained on ImageNet, and provides a powerful, yet efficient feature extraction. The U-Net architecture with its skip connections and upscaling paths is used as the decoder, which allows for precise localization.

This combination is especially suitable for applications where computational efficiency is crucial, such as mobile or embedded devices, while maintaining a high level of accuracy in the segmentation task.


## Table of Contents

- [Installation](#Installation)
- [Dataset](#Dataset)
- [Training](#Training)
- [Inference](#Inference)
- [Results](#Results)
- [To-Do](#to-do)
- [Contributing](#contributing)

## Installation
To use this implementation, you will need to have Python >= 3.9 installed on your system, as well as the following Python libraries:

```
git clone https://github.com/adityasihag1996/PortraitSeg.git
cd PortraitSeg
pip install -r requirements.txt
```

## Dataset
This model requires a dataset consisting of images and corresponding segmentation masks.\
Before training, organize your dataset in the following directory structure:

```
/dataset
    /images
        image_1.jpg
        mask_1.png
        image_2.jpg
        mask_2.png
        ...
```

Update the dataset path in the config file to point to your dataset directory.\

The following dataset (AISegment) was use for training purposes.
https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets

## Training
The training process involves fine-tuning the MobileNetV2 backbone and training the U-Net decoder.

To start the training process, adjust the parameters in `config.py` to your needs. This includes the dataset path, hyperparameters such as learning rate and batch size, and training settings such as the number of epochs.

Once you have configured the training parameters, you can start the training process by running:

```
python train.py
```

The script train.py will handle the training loop, checkpoint saving, and logging.

## Inference
For inference, use the predict.py script. This script will load a trained model and perform segmentation on the provided input images.

```
python predict.py --image path_to_image.jpg --model_path path_to_model.pth
```
The prediction will be saved in the same directory.

## Results
After training, you can find the trained model weights in the checkpoints directory. **_(COMING SOON..)_**

(IMAGE INFERENCE PATH)

## To-Do

- [ ] Boundary loss.
- [ ] Transformer based models.

## Contributing
Contributions to improve the project are welcome. Please follow these steps to contribute:

Fork the repository.\
Create a new branch for each feature or improvement.\
Submit a pull request with a comprehensive description of changes.\