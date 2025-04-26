import torch
import torch.nn as nn
import functions as f
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, inputsize):
        super().__init__()
        self.actfunc = f.ActFunc.apply
        self.fc_a = nn.Linear(inputsize, 10, bias= True)
        self.fc_b1 = nn.Linear(inputsize, 5, bias = True)
        self.fc_b2 = nn.Linear(10, 5, bias = True)
        self.fc_c1 = nn.Linear(5, 3, bias = True)
        self.fc_f = nn.Linear(inputsize, 8, bias= True)
        self.fc_c2 = nn.Linear(8, 3, bias = True)
        self.dropout = nn.Dropout(p=0.75)
    def forward(self,x):
        #input = x
        x_11 = self.actfunc(self.fc_a(x))
        x1 = self.dropout(x_11)
        #x1 = x 
        #print(x.size())
        #print(x1.size())
        x_2 = self.actfunc(self.fc_b1(x)+ self.fc_b2(x1))
        
        x_f = torch.tanh(self.fc_f(x))
        x_3 = self.actfunc(self.fc_c1(x_2) - self.fc_c2(x_f))

        return x_3

if __name__=="__main__":
    from torch.utils.data import DataLoader
    #t = torch.tensor([[2,3],[5,8]], dtype= torch.float32)
    #rb = ResidualBlock(2)
    #print('fvb')
    #model= rb(t)

    train_loader = [(torch.randn(20,2), torch.randn(20,3)) for _ in range(20)]
    #print(train_loader)
    #target = torch.randn(20,2)

    #train_tensor = [(torch.randn(20,2) for _ in range(20)]
    model= ResidualBlock(2)

    num_epoch = 5
    criteria = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    loss_list = []
    for epoch in range(num_epoch):
        model.train()
        for batch_idx, (data,target) in enumerate(train_loader):
            data = data.to(device='cpu')
            target = target.to(device='cpu')
            # forward
            scores = model(torch.tensor(data))
            print(scores.size())
            loss = criteria(scores, torch.tensor(target))
            
            # backward
            loss.backward()

            # update the weight
            optimizer.step()
        loss_list.append(loss.item())
    print(loss_list)

    




    


    
