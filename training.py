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
from sklearn.metrics import confusion_matrix
class custom_model_1(nn.Module):
    def __init__(self, output_size, hidden_layer=1024, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
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
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_layer, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, output_size)
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

    for args in testloader:
        images, labels = args[0], args[1]
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
    orig_labels = []
    pred_labels = []
    for args in test_loader:
        images, labels = args[0], args[1]
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        orig_labels.extend(labels.tolist())
        pred_labels.extend(output.max(1).indices.tolist())
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    print(confusion_matrix(orig_labels, pred_labels))
    return test_loss, accuracy
def plot_loss(train_loss, val_loss, accuracy, name):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    plt.figure()
    plt.title("Train & Val loss")
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig(f"plots/trainvVal_loss_{name}.png")
    plt.close()
 
    plt.figure()
    plt.title("Val accuracy")
    plt.plot(accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f"plots/val_accuracy_{name}.png")
    plt.close()

# Define function to save checkpoint
def save_checkpoint(model, path):
    checkpoint = {'hidden': model.n_hidden,
                  'out': model.n_out,
                  'labelsdict': model.labelsdict,
                  'state_dict': model.state_dict(),
                  'opti_state_dict': model.optimizer_state_dict,
                  }
    torch.save(checkpoint, path)

def train(train_loader,val_loader,model, optimizer,labeldict,epochs=10):
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
    # Add model info 
    model.n_hidden = 1024
    model.n_out = len(labeldict)
    model.labelsdict = labeldict
    model.optimizer_state_dict = optimizer.state_dict
    # save_checkpoint(model, "./2_classfier_model.h5py")
    plot_loss(training_loss_list, val_loss_list, val_accuracy)
    print("-- End of training --")

def train_binary(train_loader_1,val_loader_1, train_loader_2, val_loader_2, model1, model2, optimizer1, optimizer2, labeldict1, labeldict2, epochs=10):
    criterion = torch.nn.NLLLoss()

    step1=step2=0
    running_loss1=0
    running_loss2=0
    print_every=50
    device="cuda"
    model1.to(device)
    model2.to(device)
    training_loss1_list = []
    val_loss1_list = []
    val_accuracy1 = []
    training_loss2_list = []
    val_loss2_list = []
    val_accuracy2 = []
    for e in range(epochs):
        model1.train()
        model2.train()
        for image, label, _ in train_loader_1:
        #     print("-----")
            # print(k)
            # print(image.shape)
            # print(label.shape)
            image= image.to(device)
            label=label.to(device)
            optimizer1.zero_grad()
            predicted_labels = model1.forward(image)
            loss1= criterion(predicted_labels,label)
            loss1.backward()
            optimizer1.step()
            test_loss1=0
            accuracy1=0
            running_loss1+=loss1.item()
            step1+=1
            if step1%print_every==0:
                with torch.no_grad():
                    model1.eval()
                    test_loss1, accuracy1=validation( model1, val_loader_1,criterion,device)
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print("Epoch: {}/{} - ".format(e + 1, epochs),
                      " {} ".format(dt_string),
                      "Training Loss (Normal/Infected): {:.3f} - ".format(running_loss1 / print_every),
                      "Validation Loss (Normal/Infected): {:.3f} - ".format(test_loss1 / len(val_loader_1)),
                      "Validation Accuracy (Normal/Infected): {:.3f}".format(accuracy1 / len(val_loader_1)))
                training_loss1_list.append(running_loss1/print_every)
                val_loss1_list.append(test_loss1/len(val_loader_1))
                val_accuracy1.append(accuracy1/len(val_loader_1))
            running_loss1=0
            model1.train()
        
        for image, label in train_loader_2:
            image= image.to(device)
            label=label.to(device)
            optimizer2.zero_grad()
            predicted_labels = model1.forward(image)
            loss2= criterion(predicted_labels,label)
            loss2.backward()
            optimizer2.step()
            test_loss2=0
            accuracy2=0
            running_loss2+=loss2.item()
            step2+=1
            if step2%print_every==0:
                with torch.no_grad():
                    model2.eval()
                    test_loss2, accuracy2=validation( model2, val_loader_2,criterion,device)
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print("Epoch: {}/{} - ".format(e + 1, epochs),
                      " {} ".format(dt_string),
                      "Training Loss (Covid/Non-covid): {:.3f} - ".format(running_loss2 / print_every),
                      "Validation Loss (Covid/Non-covid): {:.3f} - ".format(test_loss2 / len(val_loader_2)),
                      "Validation Accuracy (Covid/Non-covid): {:.3f}".format(accuracy2 / len(val_loader_2)))
                training_loss2_list.append(running_loss2/print_every)
                val_loss2_list.append(test_loss2/len(val_loader_2))
                val_accuracy2.append(accuracy2/len(val_loader_2))
            running_loss2=0
            model2.train()
    # Add model info 
    model1.n_hidden = 1024
    model1.n_out = 2
    model1.labelsdict = labeldict1
    model1.optimizer_state_dict = optimizer1.state_dict
    model2.n_hidden = 1024
    model2.n_out = 2
    model2.labelsdict = labeldict2
    model2.optimizer_state_dict = optimizer2.state_dict
    save_checkpoint(model1, "./classfier_model_normal_infected.h5py")
    save_checkpoint(model2, "./classfier_model_covid_noncovid.h5py")
    plot_loss(training_loss1_list, val_loss1_list, val_accuracy1, "1")
    plot_loss(training_loss2_list, val_loss2_list, val_accuracy2, "2")
    print("-- End of training --")
    return model1, model2


# ld_train = Lung_Train_Dataset()
# ld_val= Lung_Val_Dataset()
# ld_test= Lung_Test_Dataset()
ld_train_1 = Lung_Dataset(types="train", data_args=0, classification="binary")
ld_val_1 = Lung_Dataset(types="val", data_args=0, classification="binary")
ld_test_1 = Lung_Dataset(types="test", data_args=0, classification="binary")
ld_train_2 = Lung_Dataset(types="train", data_args=0, classification="infected_only")
ld_val_2 = Lung_Dataset(types="val", data_args=0, classification="infected_only")
ld_test_2 = Lung_Dataset(types="test", data_args=0, classification="infected_only")
# model = Net()
model1 = model2 = custom_model_1(output_size=2)
bs_val = 40
learning_rate=0.01
train_loader_1 = DataLoader(ld_train_1, batch_size = bs_val, shuffle = True)
val_loader_1=DataLoader(ld_val_1, batch_size = 1, shuffle = True)
test_loader_1=DataLoader(ld_test_1, batch_size = 1, shuffle = True)
train_loader_2 = DataLoader(ld_train_2, batch_size = bs_val, shuffle = True)
val_loader_2=DataLoader(ld_val_2, batch_size = 1, shuffle = True)
test_loader_2=DataLoader(ld_test_2, batch_size = 1, shuffle = True)
optimizer1=torch.optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2= torch.optim.Adam(model2.parameters(), lr=learning_rate)
labeldict1 = ld_train_1.classes
labeldict2 = ld_train_2.classes
model1, model2 = train_binary(train_loader_1,val_loader_1, train_loader_2, val_loader_2, model1, model2, optimizer1, optimizer2, labeldict1, labeldict2, epochs=10)

criterion = torch.nn.NLLLoss()
test_loss1, accuracy1 = test(model1, test_loader_1,criterion,"cuda")
test_loss2, accuracy2 = test(model2, test_loader_2,criterion,"cuda")
print("Test Loss (Normal/Infected): {:.3f} - ".format(test_loss1 / len(test_loader_1)),
    "Test Accuracy (Normal/Infected): {:.3f}".format(accuracy1 / len(test_loader_1)))
print(
    "Test Loss (Covid/Non-covid): {:.3f} - ".format(test_loss2 / len(test_loader_2)),
    "Test Accuracy (Covid/Non-covid): {:.3f}".format(accuracy2 / len(test_loader_2)))