import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

# Get the data
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=24, shuffle=True)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=24, shuffle=False)


classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

def imshow(img):
  img = img /2 + 0.5 # unnormalize the image
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()

dataiter = iter(test_loader)
images, labels = next(dataiter)

imshow(images[0])

# Define the model
class MLP (nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Dimensions of the layers
input_size = 784
hidden_size = 500
output_size = 10

learning_rate = 0.001
num_epochs = 5
batch_size = 24

# Definition of the model
model = MLP(input_size, hidden_size, output_size)
model.to(device)

# Definition of the loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (The optimization algorithm, in this case SGD)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

losses = []

# Training Loop
for epoch in range(num_epochs): 
  for i, (images, labels) in enumerate(train_loader):
    # Reshape images to fit the input size
    images = images.view(-1, input_size)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (i+1) % 100 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

correct = 0 
total = 0

with torch.no_grad():
   for images, labels in test_loader:
        images = images.view(-1, input_size)  # Flatten input images
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
  
accuracy = 100 * correct / total  
print('Accuracy of the network on the 10000 test images: {} %'.format(accuracy))