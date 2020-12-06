import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from net import Net

data = np.load("train_data.npy")
labels = np.load("train_labels.npy")

class dataset(Dataset):
    def __init__(self, data, target):
        self.samples = len(data)
        # convert to tensors
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(target)
        
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

if __name__ == '__main__':
    if torch.cuda.is_available() == True:
        device = torch.device("cuda:0")
        print("running on gpu")
    else:
        device = torch.device("cpu")
        print("running on cpu")

    lr = 0.001
    batch_size = 32
    epochs = 10

    net = Net().to(device)
    ds = dataset(data, labels)
    dataloader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("training with", epochs, "epochs")
    for epoch in range(1, epochs+1):
        cnt=0
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            output = net(x.view(-1, 6))
            loss = criterion(output, torch.max(y.view(-1, 1), dim=1)[0])
            net.zero_grad()
            loss.backward()
            optimizer.step()
            cnt+=loss.item()

        print("epoch", epoch, "loss:", cnt)
    try:
        torch.save(net.state_dict(), "model.pth")
        print("saved model")

    except:
        print("failed to save model")
        
