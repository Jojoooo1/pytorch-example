import torch
import torch.nn as nn  # neural network

import torchvision.transforms as transforms
import torchvision.datasets as dsets

# torchvision consists of popular datasets, model architectures, and
# common image transformations for computer vision.

from torch.autograd import Variable

batch_size = 100

# Step 1. Load Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
print(len(train_dataset))
test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Step 2. Make Dataset Iterable
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Step 3. Create Model Class
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


# Step 4. Instantiate Model Class
# batch_size = 100
# total data = 60 000
n_iters = 3000
epochs = n_iters / (len(train_dataset) / batch_size)
input_dim = 784
output_dim = 10
lr_rate = 0.001
model = LogisticRegression(input_dim, output_dim)

# Step 5. Instantiate Loss Class
criterion = torch.nn.CrossEntropyLoss(
)  # computes softmax and then the cross entropy
# Step 6. Instantiate Optimizer Class
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
# Step 7. Train Model
iter = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28 * 28))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            print("Iteration: {}. Loss: {}. Accuracy: {}.".format(
                iter, loss.item(), accuracy))
