# Importing Dependencies
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, Subset
from dataset import CustomDataset
import torchvision
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
# Creating transforms for training and validation datasets
train_ts = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.GaussianBlur(7, 1), # Using Gaussian blur to reduce noise in the scaleograms
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

val_ts = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# Creating the Dataset
dataset = ImageFolder('/Users/yash/Downloads/DatasetCWT')
# train_ds, val_ds = random_split(dataset, [0.75, 0.25])
# len_ts = len(train_ds)
# len_vs = len(val_ds)
# train_ds = CustomDataset(train_ds, transform= train_ts)
# val_ds = CustomDataset(val_ds, transform= val_ts)
# train_dl = DataLoader(train_ds, batch_size= 16, num_workers=0, pin_memory= True, shuffle = True)
# val_dl = DataLoader(val_ds, batch_size= 16, num_workers=0, pin_memory= True, shuffle = True)

# Some Helper functions
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


# Using tensorboard to analyse the results
def write_to_tb(writer, t_loss, t_acc, v_acc, fold, tb_step):
    writer.add_scalar(tag = f"{fold}Training Loss", scalar_value= t_loss, global_step= tb_step)
    writer.add_scalar(tag=f"{fold}Training Accuracy", scalar_value=t_acc, global_step=tb_step)
    writer.add_scalar(tag=f"{fold}Validation Accuracy", scalar_value=v_acc, global_step=tb_step)

def confusion_mat(actual, pred):
    actual = np.concatenate(actual).ravel()
    pred = np.concatenate(pred).ravel()
    matrix = confusion_matrix(actual, pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    cm_disp = ConfusionMatrixDisplay(confusion_matrix= matrix, display_labels= ['HREOG', 'HLEOG', 'VUPEO', 'VDEOG', 'Blink'])
    cm_disp.plot()
    plt.show()
# Creating the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Using alexnet pretrained on ImageNet
        self.alex = torchvision.models.alexnet(pretrained = True)
        for params in self.alex.parameters():
            params.requires_grad = False
        self.alex.classifier[6] = nn.Sequential(
            nn.Linear(4096, 2000),
            nn.ReLU(),
            #nn.Dropout(p = 0.2),
            nn.Linear(2000, 500),
            nn.ReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )
        # self.alex.classifier[6] = nn.Sequential(
        #     nn.Linear(4096, 1000),
        #     nn.ReLU(),
        #     nn.Linear(1000, 5)
        # )

        # Only using the feature extractor of alexnet. The classifier is free to learn
        for params in self.alex.classifier.parameters():
            params.requires_grad = True
        self.softmax = nn.Softmax()

        # Setting a CNN baseline
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(3872, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 5)
        )
    def forward(self, x):
        x = self.alex(x)
        x = self.softmax(x)
        return x


# Fit function
def fit(model, train_dl, val_dl, opt, loss_fn, len_ts, len_vs, writer, epoch, fold, epochs):

    # Training Phase
    model.train()
    train_cr = 0
    history = []
    val_history = []
    model.train()
    for batch in tqdm(train_dl):
        data, labels = batch
        out = model(data)
        loss = loss_fn(out, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Validation Phase
    model.eval()
    label_acc =[]
    pred_acc = []
    with torch.no_grad():
        history.append(loss.item())
        #preds = torch.argmax(out, dim = 1)
        #train_cr += (preds == labels).sum()
        val_cr = 0
        for batch in val_dl:
            data, labels = batch
            out = model(data)
            preds = torch.argmax(out, dim=1)
            pred_acc.append(preds.cpu().numpy())
            label_acc.append(labels.cpu().numpy())
            val_cr += (preds == labels).sum()
            val_acc = val_cr/float(len_vs)
            val_history.append(val_acc)
        if epoch == epochs -1:
            confusion_mat(label_acc, pred_acc)
    print(f"Loss: {np.mean(history)}\nTrain_Acc: {train_cr/float(len_ts)}\nVal_Acc: {val_acc}")
    write_to_tb(writer, np.mean(history), train_cr/float(len_ts), val_acc, fold, epoch)
def main():
    # Hyperparameters
    writer = SummaryWriter("./TB_Logs")

    # Implementing KFold Cross-Validation due to limited size of dataset
    kfold = KFold(n_splits= 4, shuffle= True)
    for fold, (tr_id, val_id) in enumerate(kfold.split(dataset)):
        tb_step = 0
        train_ds = CustomDataset(Subset(dataset, tr_id), transform= train_ts)
        val_ds = CustomDataset(Subset(dataset, val_id), transform= val_ts)
        len_ts = len(train_ds)
        len_vs = len(val_ds)
        train_dl = DataLoader(train_ds, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)
        device = 'mps'
        train_dl = DeviceDataLoader(train_dl, device)
        val_dl = DeviceDataLoader(val_dl, device)
        model = torchvision.models.alexnet(pretrained=True)

        # Hyperparameters
        epochs = 120
        lr = 1e-5
        weight_decay = 1e-4
        model = Model()
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay= weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        print(f"FOLD: {fold}")

        # Training Loop
        for epoch in range(epochs):
            fit(model, train_dl, val_dl, optimizer, loss_fn, len_ts, len_vs, writer, epoch, fold, epochs)
        torch.save(model.state_dict(), './checkpoint1.pth')
if __name__ == '__main__':
    main()
