from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

batch_size = 24

# 1. Get the data
train_set = torchvision.datasets.MNIST(
    root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.MNIST(
    root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 2. visualize the data
image, label = train_set[0]
image = image.numpy()

# plt.imshow(image[0], cmap='gray')
# plt.show()

# Feature maps: Represents the result of applying a set of filters (kernels) to an input image.

# 3. Create the model class

class CNN(nn.Module):
    def __init__(self):
        # This is used to call the constructor of the parent class
        super(CNN, self).__init__()
        # Conv Layers
        # A convolutional layer is a layer that moves in matrix and applies a dot product.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Used to down sample our feature maps. Keep the most important features and remove the rest.
        # Stride: How many pixels we move the kernel each time.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=3)

        self.fc1 = nn.Linear(2592, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten the output for the fc layers
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# 4. Define the hyper parameters
epochs = 20
learning_rate = 0.001

# Init the model
model = CNN()

# Cross Entropy Loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

losses = []
total_step = len(train_loader)

checkpoint_interval= 2 
checkpoints_dir = "checkpoints/"

# 5. Run the training lopp
for epoch in range(epochs):
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
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

    if (epoch+1) % checkpoint_interval == 0:
          torch.save(model.state_dict(), checkpoints_dir + "mnist-cnn-epoch-{}.pt".format(epoch+1))
          print("Checkpoint saved")


plt.plot(losses)
plt.show()

# 6. Test the model
accurate = 0
total = 0

for i, (images, labels) in test_loader:
    output = model(images)
    _, predicted = torch.max(output.data, 1)
    accurate += (predicted == labels).sum().item()
    total += labels.size(0)

print("Accuracy: ", accurate/total *100)