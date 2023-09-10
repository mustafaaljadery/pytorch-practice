import torch
import torchvision
import torch.nn as nn
from torch.data.utils import DataLoader

train_data = torchvision.datasets.CIFAR100(
    root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=24, shuffle=True)

test_data = torchvision.datasets.CIFAR100(
    root="./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(test_data, batch_size=24, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0])
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplaces != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes,
                          out_channels=planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


num_classes = 100
num_epochs = 20

model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 20
running_losses = []

# Train the model
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:
            print(f"Epoch: {epoch+1}, Step: {i+1}, Loss: {running_loss/10}")
            running_losses.append(running_loss/10)
            running_loss = 0.0

# Test the model
total = 0
correct = 0

with torch.no_grad():
    for (inputs, labels) in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
