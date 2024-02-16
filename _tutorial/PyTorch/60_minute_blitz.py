#%% [markdown]
# # PyTorch Blitz
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
#
# Date created: 20190728

#%% [markdown]
# ## What is PyTorch?
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

#%%
from __future__ import print_function
import torch


#%%
x = torch.empty(5,3)
print(x)

#%%
x = torch.rand(5,3)
print(x)

#%%
x=torch.zeros(5,3,dtype=torch.long)
print(x)

#%%
x=torch.tensor([5.5,3])
print(x)

#%%
x = x.new_ones(5,3,dtype=torch.double) # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float) # override dtype!
print(x)

#%%
print(x.size())

#%%
y = torch.rand(5,3)
print(x+y)

#%%
print(torch.add(x,y))

#%%
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

#%%
y.add_(x)  # _ denotes that the tensor y is mutated in-place of the 
print(y)

#%%
print(x[:,1])

#%%
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)  # -1 causes view to infer from other dims
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

#%%
a = torch.ones(5)
print(a)

#%%
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

#%%
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

#%%
# Try CUDA: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#cuda-tensors
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
else:
    print("CUDA unavailable.")


#%% [markdown]
# ## Autograd: Automatic Differentiation
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

#%%
import torch

#%%
x = torch.ones(2,2, requires_grad=True)
print(x)

#%%
y = x + 2
print(y)

#%%
print(y.grad_fn)

#%%
z = y * y * 3
out = z.mean()
print(z, out)

#%%
a = torch.randn(2,2)
a = ((a*3) / (a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)


#%%
out.backward()

#%%
print(x.grad)

#%% [markdown]
# #### "Generally speaking, torch.autograd is an engine for computing vector-Jacobian product. That is, given any vector v=(v1v2⋯vm)T, compute the product vT⋅J."

#%%
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

#%%
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

#%%
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print( (x**2).requires_grad )

#%% [markdown]
# ## Neural Networks
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

#%%
# Feed-forward convolutional NN for MNIST digits classification.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 sq convolution
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        # affine op: y = Wx + b
        self.fc1 = nn.Linear(16*6*6,120) # 6*6 from dims of image
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # If the size is a square, then only a single number can be specified.
        x = F.max_pool2d( F.relu(self.conv2(x)), 2 )
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dims except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

#%%
# print the learnable params
params = list(net.parameters())
print(len(params))
print(params[0].size())

#%%
# random input image
input = torch.randn(1,1,32,32)
out = net(input)
print(out)

#%%
# zero-out the gradient buffers with random values.
net.zero_grad()
out.backward(torch.randn(1,10))


#%%
# Loss function
output = net(input)
target = torch.randn(10)    # dummy target
target = target.view(1,-1)  # make the target the same shape as the output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

#%%
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

#%%
net.zero_grad() # zeros-out the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


#%%
# implement SGD (supposedly - where is the stochastic term???)
learning_rate = 0.01

for f in net.parameters():
    f.data.sub_( f.grad.data * learning_rate )


#%%
# or use a built-in optimizer
import torch.optim as optim

optimizer = optim.SGD( net.parameters(), lr = learning_rate)

# in training loop:
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # performs update

#%%
print(loss)
print(output)

#%% [markdown]
# ## Training a classifier
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

#%%
import torch
import torchvision
import torchvision.transforms as transforms
import pathlib

#%%
transform = transforms.Compose(
    [   transforms.ToTensor(),
        transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5) ) ] )

data_root_path = str(pathlib.Path.home()) + "/Documents/syncable/home/dev/data_science/practice/PyTorch/data"

trainset    = torchvision.datasets.CIFAR10( root=data_root_path, train=True, download=True, transform=transform )
trainloader = torch.utils.data.DataLoader( trainset, batch_size=4, shuffle=True, num_workers=0 )  # num_workers=2 until Python3.11 RuntimeError; changed to 0, see: https://stackoverflow.com/a/66845956

testset    = torchvision.datasets.CIFAR10( root=data_root_path, train=False, download=True, transform=transform )
testloader = torch.utils.data.DataLoader( testset, batch_size=4, shuffle=False, num_workers=0 )  # num_workers=2 until Python3.11 RuntimeError; changed to 0, see: https://stackoverflow.com/a/66845956

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow( np.transpose(npimg, (1,2,0) ) )
    plt.show()

# get random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow( torchvision.utils.make_grid(images) )
# print labels
print( " ".join("%5s" % classes[labels[j]] for j in range(4) ) )

#%%
import os
os.getcwd()

#%% [markdown]
# ## Define a conv net -- same as before, but taking 3-channel (not 1-channel) images as input.
#
# (note: copy-pasted, not manually transcribed for practice)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#%%
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%%
outputs = net(images)

#%%
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#%%
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#%%
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

#%%
