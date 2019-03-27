import torch
import torch.nn.functional as F

class Net1(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net1, self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.out=torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.out(x)
        return x

net1=Net1(1,10,1)
net2=torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
)

print(net1)
print(net2)
