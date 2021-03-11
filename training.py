# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime
from data_loader import *
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.fc1 = nn.Linear(87616, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim = 1)
        return output
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy



def train(train_loader,val_loader,model, optimizer,epochs=10):
    criterion = torch.nn.NLLLoss()


    steps=0
    running_loss=0
    print_every=50
    device="cuda"
    model.to(device)
    for e in range(epochs):
        model.train()
        for k, (image, label) in enumerate(train_loader):
        #     print("-----")
        #     print(k)
        #     print(v[0])
        #     print(v[1]
            image= image.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            predicted_labels = model.forward(image)
            loss= criterion(predicted_labels,label)
            loss.backward()
            optimizer.step()
            test_loss=0
            accuracy=0
            running_loss+=loss.item()
            steps+=1
            if steps%print_every==0:
                with torch.no_grad():
                    model.eval()
                    test_loss, accuracy=validation( model, val_loader,criterion,device)
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print("Epoch: {}/{} - ".format(e + 1, epochs),
                      " {} ".format(dt_string),
                      "Training Loss: {:.3f} - ".format(running_loss / print_every),
                      "Validation Loss: {:.3f} - ".format(test_loss / len(val_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(val_loader)))
            running_loss=0
            model.train()
ld_train = Lung_Train_Dataset()
ld_val= Lung_Val_Dataset()
model = Net()
bs_val = 40
learning_rate=0.01
train_loader = DataLoader(ld_train, batch_size = bs_val, shuffle = True)
val_loader=DataLoader(ld_val, batch_size = 1, shuffle = True)
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
train(train_loader,val_loader,model, optimizer,epochs=10)

