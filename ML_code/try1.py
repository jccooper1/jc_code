import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
class Model(nn.Module):    
    def __init__(self):        
        super(Model, self).__init__()        
        self.f1 = nn.Linear(28*28, 100, bias=True)        
        self.f2 = nn.Linear(100, 10, bias=True)    
        def forward(self, data):        
            data = self.f1(data)        
            data = F.relu(data)        
            data = self.f2(data)        
            return data
root = 'data/MNIST'
transformfun = transforms.Compose([    transforms.ToTensor(),    ])
mnist = torchvision.datasets.MNIST(root, train=True, transform=transformfun, target_transform=None, download=True) 
count = 8
mnist_loader = torch.utils.data.DataLoader(dataset = mnist    ,batch_size=count    ,shuffle=True) 
def other_optimizer(optimizer,model,mnist_loader):    
    for epoch in range(10):  #反向传播10次，并计算总的损失值        
        total_loss = 0        
        for batch in mnist_loader:            
            images, labels = batch             
            inputs = images.reshape(count,-1)            
            output = model(inputs)            
            loss = F.cross_entropy(output, labels)#计算损失值            
            optimizer.zero_grad()            
            loss.backward() # 反向传播            
            optimizer.step() # 更新权值            
            total_loss += loss.item()        
            print("反向传播次数：", epoch, "  loss:", total_loss)
model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)
other_optimizer(optimizer,model,mnist_loader)

optimizer_Nesterov=optim.SGD(model.parameters(),lr=0.01, momentum=0.9, nesterov=True)
other_optimizer(optimizer_Nesterov,model,mnist_loader)