import torchvision.datasets as datasets
import os
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
from variables import device
import torch


def get_mnist_datasets(train_save_dir, test_save_dir, transform=None):
    mnist_dataset_train = datasets.MNIST(root=train_save_dir, train=True, download=True, transform=transform)
    mnist_dataset_test = datasets.MNIST(root=test_save_dir, train=False, download=True, transform=transform)
    return mnist_dataset_train, mnist_dataset_test


def download_mnist_dataset():
    # Define the directory to save the MNIST images
    train_save_dir = os.path.join('mnist_images', "train")
    test_save_dir = os.path.join('mnist_images', "test")
    # Delete the directory if it exists
    for path in [train_save_dir, test_save_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(train_save_dir)
    os.makedirs(test_save_dir)

    # Download the MNIST dataset
    mnist_dataset_train, mnist_dataset_test = get_mnist_datasets(train_save_dir, test_save_dir)

    # Save the images locally - Train set
    for i, (image, label) in tqdm(enumerate(mnist_dataset_train)):
        image.save(os.path.join(train_save_dir, f'mnist_{i}.png'))

    # Save the images locally - Test set
    for i, (image, label) in tqdm(enumerate(mnist_dataset_test)):
        image.save(os.path.join(test_save_dir, f'mnist_{i}.png'))

    return train_save_dir, test_save_dir


def preprocess(image_paths):
    images = []
    for image_path in image_paths:
        pil_image = Image.open(image_path)
        image = torch.Tensor(np.array(pil_image))
        images.append(image.reshape(28 * 28).to(device))
    return torch.stack(images)
