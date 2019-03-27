import torch
import a as Data

torch.manual_seed(1)
BATCH_SIZE=5
x=torch.linspace(1,10,10) # 1,2,3,...10
y=torch.linspace(10,1,10) #10,9,8.....1

torch_dataset=Data.TensorDataset(x,y) #sample data,target/label data
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

def show_batch():
    for epoch in range(3):
        for step,(batch_x,batch_y) in enumerate(loader):
            # train data
            print('Epoch:',epoch,'|Step:',step,'|batch x:',batch_x.numpy(),'batch y:',batch_y.numpy())

show_batch()