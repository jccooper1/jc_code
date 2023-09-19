#二阶常微分方程，dy2/dx2+(1/5)*(dy/dx)+y+(1/5)exp(-x/5)cos(x)=0
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# same_seed 确定结果一致
def same_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# 初值
g0 = 0
g0_diff = 1
x_real = torch.linspace(0, 1, 500).view(-1,1)
y_real = torch.exp(-x_real/5) * torch.sin(x_real)

x = x_real.numpy()
y = y_real.numpy()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

delta_x = 1e-10000

# 神经网络的构造函数
def predict(model, x_real, g0_diff, g0):
    train_x = x_real.to(device)
    u_diff_pred = model(train_x) * train_x ** 2 + g0_diff * train_x + g0
    return u_diff_pred
class Basic_Unit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Basic_Unit, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
            
        )

    def forward(self, x):
        return self.layers(x)
class Neuron_fun(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
        super(Neuron_fun, self).__init__()

        self.layers = nn.Sequential(
            Basic_Unit(input_dim, hidden_dim),

            *[Basic_Unit(hidden_dim, hidden_dim) for _ in range(hidden_layers)],

            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)



criterion = nn.MSELoss()



def train(model, lr, iterations, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0, last_epoch=-1)
    #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    model.train()
    tqbar = tqdm(range(iterations), ncols=100)
    costs = []
    best_loss = math.inf

    for epoch in tqbar:
        tqbar.set_description(f"{epoch+1}/{iterations}")
        #train_x, train_y = x_real.to(device), y_real.to(device)
        train_x = x_real.to(device)
        train_y = -torch.exp(-train_x/5) / 5 * torch.cos(train_x)


        y_nn = predict(model, train_x, g0_diff, g0)
        y_nn_diff = (predict(model, train_x + delta_x, g0_diff, g0) - predict(model, train_x, g0_diff, g0)) / delta_x
        y_nn_diff2 = (predict(model, train_x, g0_diff, g0) - 2 * predict(model, train_x + delta_x, g0_diff, g0) + predict(model, train_x + 2 * delta_x, g0_diff, g0) ) / (delta_x**2)

        y_pred = y_nn_diff2 + y_nn_diff / 5 + y_nn


        loss = criterion(y_pred, train_y)
        costs.append(loss.item())

        tqbar.set_postfix({"lr":optimizer.param_groups[0]["lr"],"loss":loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if best_loss > loss.item():
            best_loss = loss.item()
            torch.save(model.state_dict(), "./model.ckpt")


    return model, costs

same_seed(2048)

model = Neuron_fun(input_dim=1, hidden_dim=512, output_dim=1, hidden_layers=2).to(device)
#print(model)
lr = 1e-3
iterations = 100
weight_decay = 0

model, costs = train(model, lr, iterations, weight_decay)

model.load_state_dict(torch.load("./model.ckpt"))


model.eval()
with torch.no_grad():
    pred_y = predict(model, x_real, g0_diff=1, g0 = 0).cpu().numpy()
    plt.figure(0)
    plt.plot(x, y, "g-")
    plt.plot(x, pred_y, "r-")
    plt.legend(["real", "pred"])
    plt.xlabel("x")
    plt.ylabel("y")


