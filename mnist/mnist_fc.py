import torch
import torch.nn as nn
import torch.utils.data as DataLoader
from torchvision import datasets, transforms
torch.manual_seed(0)
batch_size = 64
learning_rate = 0.01
epochs = 5

# MNIST Dataset
transform=transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Data Loader
train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Simple Neural Network Model
model=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)#or torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    model.train()
    total_loss=0
    for x,y in train_loader:
        #foward
        logits=model(x)
        loss=criterion(logits,y)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        print(f'Epoch {epoch}, Loss: {(total_loss/len(train_loader)):.4f}')

# Test the model
model.eval()
correct=0
total=0
with torch.no_grad():
    for x,y in test_loader:
        logits=model(x)
        pred=logits.argmax(dim=1)
        correct+=(pred==y).sum().item()
        total+=y.size(0)
print(f'Test Accuracy: {(correct/total)*100:.2f}%')

#final results
#Test Accuracy: SGD:91.75%
#Test Accuracy: Adam:96.54%

# The reason why SGD has a lower accuracy than Adam is because
# Adam is an adaptive learning rate optimizer,
# it adapts the learning rate for each parameter based on the gradient
# which can lead to faster convergence and better performance.
# SGD uses a fixed learning rate, which may not be optimal for the problem at hand.
