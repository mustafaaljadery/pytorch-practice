import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

classes = ["airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"]

# Get the data
# num of workers = the number of subprocesses to run the data.
train_data = torchvision.datasets.CIFAR10(
    root="./data2", train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(
    train_data, batch_size=24, shuffle=True, device="cuda")

test_data = torchvision.datasets.CIFAR10(
    root="./data2", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(
    test_data, batch_size=24, shuffle=False, device="cuda"
)
# Visualize the data

item = train_data[2]
image, label = item

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
      x = self.conv1(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.conv2(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.conv3(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.conv4(x)
      x = self.relu(x)
      x = self.conv5(x)
      x = self.relu(x)
      
      # Flatten: Convert a multi-dimensional tensor into a 1D tensor.
      x = x.view(x.size(0), -1)
      
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      
      return x

epochs = 20 
learning_rate = 0.001
    
model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
running_losses = []
total_step = len(train_loader)

checkpoint_interval = 2
checkpoint_dir = "checkpoints/cifar10/"

for epoch in range(epochs):
  # Cumulativee or running sum of loss values computed during the epoch. It's a way to keep track of the average loss value per epoch.
  running_loss = 0.0

  for i, (inputs, labels) in enumerate(train_loader):
      
      optimizer.zero_grad()
      
      outputs = model(inputs)
      
      loss = criterion(outputs, labels)
      
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()
      
      if (i+1) % 10 == 0:
        losses.append(loss.item())
      
      if (i+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")
  
  if (epoch+1) % checkpoint_interval == 0:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        'losses': losses
        }, checkpoint_dir + f"cifar10-cnn-{epoch+1}.ckpt")
    print("Checkpoint saved")

  running_losses.append(running_loss / total_step) 
  
plt.plot(running_losses)
plt.show()
    
accurate = 0
total = 0

with torch.no_grad():
  for inputs, labels in test_loader:
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    accurate += (predicted == labels).sum().item()
    
print(f"Accuracy of the model on the 10000 test images: {100 * accurate / total}%")
