import os
from pathlib import Path
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, Normalize

from model import VGG11

path_to_dataset = Path(os.getcwd()) / "dataset"
path_to_dataset.mkdir(exist_ok=True)

def grayscale_to_rgb(image):
    if image.shape == torch.Size([3, 224, 224]):
      return image
    return torch.cat([image, image, image], dim=0)
  
transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    grayscale_to_rgb,
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

  ]
)

def train(model, optimizer, loss_function):
  for _ in (t:=trange(EPOCHS)):
    for _, data in enumerate(train_data, 0):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)
      # labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)

      optimizer.zero_grad()
      
      outputs = model(inputs)

      # logits = F.softmax(outputs, dim=1)
      logits = outputs
      
      loss = loss_function(logits, labels)
      loss.backward()
      optimizer.step()
      
      t.set_description(f"Running loss {(loss.item()):.4f}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = datasets.Caltech101(root=path_to_dataset, 
                          download=True, 
                          transform=transforms)

BATCH_SIZE = 16
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 64
EPOCHS = 2
LR = 1e-2

train_data, test_data = torch.utils.data.random_split(data, [.8, .2])
train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
  model = VGG11(input_channels=INPUT_CHANNELS,
                out_channels=OUTPUT_CHANNELS).to(device)

  total_params = sum(p.numel() for p in model.parameters())
  print(f"[INFO]: {total_params:,} total parameters.")

  optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
  # optimizer = optim.Adam(model.parameters(), lr=LR)

  loss_function = nn.CrossEntropyLoss()
  
  train(model, optimizer, loss_function)

