import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from tqdm import tqdm, TqdmLogger as logger


class LinReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 1)

    def forward(self, x):
        return self.lin(x)


net = LinReg()
opt = torch.optim.Adam(net.parameters())

SIZE = 10 ** 6
Xdata = torch.randn(SIZE, 10)
ydata = Xdata.sum(1) + 1

dataset = torch.utils.data.DataLoader(list(zip(Xdata, ydata)), batch_size=32, shuffle=True)
smooth_loss = 0.

for X, y in tqdm(dataset):
    opt.zero_grad()
    yp = net(X)
    loss = F.mse_loss(y, yp)
    smooth_loss = smooth_loss * .999 + loss.item() * .001
    logger.log("Loss: {:.3} | Smooth loss: {:.3}".format(loss.item(), smooth_loss))
    loss.backward()
    opt.step()
