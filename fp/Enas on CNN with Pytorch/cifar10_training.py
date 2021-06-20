import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#

class Cifar10Trainer:

    def __init__(self, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('what is the device? cpu or gpu ? ', self.device)
        #
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.48216, 0.44653),
                                  (0.24703, 0.24349, 0.26159))])
        #
        indices = np.arange(50000)
        np.random.shuffle(indices)
        #
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                       shuffle=False, num_workers=2,
                                                       sampler=torch.utils.data.SubsetRandomSampler(indices[:45000]))
        self.valloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                     shuffle=False, num_workers=2,
                                                     sampler=torch.utils.data.SubsetRandomSampler(indices[45000:50000]))

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                      shuffle=False, num_workers=2)

    def train(self, net, no_epoch=2):
        n = 0
        train_time_list = []
        net.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=3e-4)

        for epoch in range(no_epoch):
            start = time.time()
            net.to(self.device)
            net.train()
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            correct, total = 0, 0
            predictions = []
            net.eval()
            zzz = 0
            with torch.no_grad():
                for i, data in enumerate(self.valloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    xxx = time.time()
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.append(outputs)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                val_accuracy = 100 * correct / total
                yyy = time.time()
                zzz += (yyy - xxx)
                print('step number ', n)
                print('The validation set accuracy of the network is: %.2f %%' % (val_accuracy))
            print('zzz: %.2f second' % zzz)
            end = time.time()
            n += 1
            train_time = end - start
            print('The training time for this epoch is: %.2f second' % train_time)
            train_time_list.append(train_time)
        torch.save(net.state_dict(), 'weights.pth')
        return val_accuracy, train_time_list