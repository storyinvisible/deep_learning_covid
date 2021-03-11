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
import os
class custom_model_1(nn.Module):
    def __init__(self, output_size, hidden_layer=1024, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, bias=False),
                           nn.ReLU(),
                           nn.BatchNorm2d(16),
                           nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, bias=False),
                           nn.ReLU(),
                           nn.BatchNorm2d(32),
                           nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, bias=False),
                           nn.ReLU(),
                           nn.BatchNorm2d(64),
                           nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=False),
                           nn.ReLU(),
                           nn.BatchNorm2d(64),
                           nn.MaxPool2d(2, 2))
        
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5184, hidden_layer),
            nn.Dropout(p=drop_p),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_layer, output_size)
        )
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out(x)
        
        return F.log_softmax(x, dim=1)

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
def test(model,test_loader,criterion,device):
    test_loss = 0
    accuracy = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy
def plot_loss(train_loss, val_loss, accuracy):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    plt.figure()
    plt.title("Train & Val loss")
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig(f"plots/trainvVal_loss.png")
    plt.close()
 
    plt.figure()
    plt.title("Val accuracy")
    plt.plot(accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f"plots/val_accuracy.png")
    plt.close()

# Define function to save checkpoint
def save_checkpoint(model, path):
    checkpoint = {'input': model.n_in,
                  'hidden': model.n_hidden,
                  'out': model.n_out,
                  'labelsdict': model.labelsdict,
                  'lr': model.lr,
                  'state_dict': model.state_dict(),
                  'opti_state_dict': model.optimizer_state_dict,
                  'class_to_idx': model.class_to_idx
                  }
    torch.save(checkpoint, path)

def train(train_loader,test_loader,val_loader,model, optimizer,epochs=10):
    criterion = torch.nn.NLLLoss()

    steps=0
    running_loss=0
    print_every=50
    device="cuda"
    model.to(device)
    training_loss_list = []
    val_loss_list = []
    val_accuracy = []
    for e in range(epochs):
        model.train()
        for k, (image, label) in enumerate(train_loader):
        #     print("-----")
            # print(k)
            # print(image[0].shape)
            # print(label.shape)
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
                training_loss_list.append(running_loss/print_every)
                val_loss_list.append(test_loss/len(val_loader))
                val_accuracy.append(accuracy/len(val_loader))
            running_loss=0
            model.train()
    model.eval()
    test_loss, accuracy= test(model,test_loader,criterion,device)
    print("Test Loss  : {:.3f}  Test Accuracy : {:.3f} ".format(test_loss/len(test_loader), accuracy/len(test_loader)))
    plot_loss(training_loss_list, val_loss_list, val_accuracy)
    print("-- End of training --")


ld_train = Lung_Train_Dataset()
ld_val= Lung_Val_Dataset()
ld_test= Lung_Test_Dataset()
# model = Net()
model = custom_model_1(len(ld_train.classes))
bs_val = 40
learning_rate=0.01
train_loader = DataLoader(ld_train, batch_size = bs_val, shuffle = True)
val_loader=DataLoader(ld_val, batch_size = 1, shuffle = True)
test_loader=DataLoader(ld_test, batch_size = 1, shuffle = True)
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
train(train_loader,test_loader,val_loader,model, optimizer,epochs=10)

