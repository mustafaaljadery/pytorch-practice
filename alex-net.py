"""
AlexNet Notes:
- Dataset: 1.2 million high-res images, 1000 classes
- 60 million parameters, and 650,000 neurons.
- 5 convolutional layers, some of which are followed by max-pooling layers.
- 3 fully connected layes with a final 1000-way softmax.
- To reduce overfitting in th fully-connected layers they use dropout.

- CNNs to a feed-forward neural network.
- Images are 256 x 256
- Contains 8 learned layers - 5 convolutional and 3 fully-connected.
- Networks with the ReLU function train several times faster than tanh networks.
- Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map.
- Local Response Normalization: Technique used in neural networks to normalize  the activations of neurons within a specific "local" or neighboring region.
- Typically you want to apply the normalization before the activation function.

CNNs
- Filters: Weights in the CNNs. It's a matrix.
- Stride: How many pixels we move each time we slide the filter over the image.
- Padding: Add zeros to the outside of the image.
- Channels: The number of filters we use.
"""

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    normalize
])

# Load the data
train_dataset = torchvision.datasets.ImageNet(
    root='./data3', train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=24, shuffle=True, num_workers=2
)

test_dataset = torchvision.datasets.ImageNet(
    root='./data3', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=24, shuffle=True, num_workers=2
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, padding=1, stride=1, kernel_size=3)
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, padding=1, stride=1, kernel_size=3
        )

        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.local_norm = nn.LocalResponseNorm(
            size=5, alpha=0.0001, beta=0.75, k=2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.local_norm(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.local_norm(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # flatten the output
        x = x.view(-1, 256 * 6 * 6)

        # Fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
        
# Define and train the model
model = CNN().to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 90

running_losses = []

for epoch in range(epochs): 
  running_loss = 0.0

  for i, (inputs, labels) in enumerate(train_loader): 
    inputs, labels = inputs.to("cuda"), labels.to("cuda")
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()

    if (i + 1) % 100 == 0:
      print(f'Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item()}')

  running_losses.append(running_loss)
  print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# Visualize the loss
plt.plot(running_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# Evaluate the model

accurate = 0
total = 0 

with torch.no_grad():
  for (inputs, labels) in test_loader:
    inputs, labels = inputs.to("cuda"), labels.to("cuda")
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    accurate += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * accurate / total}%')
