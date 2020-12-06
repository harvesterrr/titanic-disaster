import torch
from net import Net
import numpy as np

data = np.load("test_data.npy")

if __name__ == '__main__':
    if torch.cuda.is_available() == True:
        device = torch.device("cuda:0")
        print("running on gpu")
    else:
        device = torch.device("cpu")
        print("running on cpu")

    net = Net().to(device)
    net.load_state_dict(torch.load("model.pth"))
    net.eval()
    data = torch.from_numpy(data)
    total = 0
    correct = 0
    for i, x in enumerate(data):
        x = x.to(device)
        output = net(x.view(-1, 6))
        prediction = torch.argmax(output)
        print(prediction.item())
        
