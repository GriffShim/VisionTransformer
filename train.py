import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import einops
from einops import rearrange

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for epoch in range(20):
    running_loss = 0.0
    model.eval()
    img = ddpm_obj.p_sample(0, w=2)[0]
    img = tensor_to_uint8_numpy(img)
    img = Image.fromarray(img)
    display(img)
    model.train()

    for idx, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()

        t = torch.randint(0, 400, (images.shape[0],))
        mask = torch.rand(len(labels)) < 0.05
        labels[mask] = 10
        labels = labels.to('cuda')

        x_t, eps = ddpm_obj.q_sample(images, t)
        x_t = x_t.to('cuda')
        eps = eps.to('cuda')
        t = t.to('cuda')
        outputs = model(x_t, t, labels)
        loss = (outputs - eps).square().mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if idx%100 == 0:
          print(running_loss / (idx+1))

    epoch_loss = running_loss / len(trainloader)
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}')
