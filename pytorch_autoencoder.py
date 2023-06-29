import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from model_definition import AutoEncoder
from general_utils import get_mnist_datasets
from variables import device
import os

# Hyperparameters
epochs = 5
batch_size = 128
lr = 0.001

# Download the mnist dataset
train_save_dir = os.path.join('mnist_images', "train")
test_save_dir = os.path.join('mnist_images', "test")

train_dataset, test_dataset = get_mnist_datasets(train_save_dir, test_save_dir, transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Initialize the autoencoder
model = AutoEncoder().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training
total_step = len(train_loader)
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        # Flatten the images
        images = images.reshape(-1, 28*28).to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 300 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step, loss.item()))

# Testing
with torch.no_grad():
    for images, _ in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        outputs = model(images)
        break

torch.save(model.state_dict(), "model.pt")