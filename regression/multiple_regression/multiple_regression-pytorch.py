#%%[markdown]
## Multiple linear regression in PyTorch
#
# Date created: 20240222
#
# Src: [Training a Single Output Multilinear Regression Model in PyTorch - MachineLearningMastery.com](https://machinelearningmastery.com/training-a-single-output-multilinear-regression-model-in-pytorch/)

#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)

#%%
class Data(Dataset):
    # ctor
    def __init__(self):
        ### data X
        
        # original ==> w/ batch_size=2, loss decreases from 7 -> ~0.1, w/ a spike at ~10. loss curve smooths-out as batch_size -> full-batch of 40.
        # self.x = torch.zeros(40, 2)
        # self.x[:, 0] = torch.arange(-2, 2 ,0.1)
        # self.x[:, 1] = torch.arange(-2, 2, 0.1)

        # 5x samples, and a different vec ==> w/ batch_size=2, loss decreases from 6 -> ~0.1, but spikes periodically10. loss curve smooths-out as batch_size -> full-batch of 200.
        self.x = torch.zeros(200, 3)
        self.x[:, 0] = torch.arange(-1, 1, 0.01)
        self.x[:, 1] = torch.arange(-1*2, 1*2, 0.01*2)
        self.x[:, 2] = torch.arange(-1, 1, 0.01)

        # 25x samples, all different vecs ==> w/ batch_size=2, loss = 0 until >= iterations=5. loss curve does not smooth-out even as batch_size -> full-batch of 1000.
        # self.x = torch.zeros(1000, 4)
        # self.x[:, 0] = torch.arange(-1*5, 1*5, 0.01)
        # self.x[:, 1] = torch.arange(-1.2*5, 1.2*5, 0.012)
        # self.x[:, 2] = torch.arange(-1*50, 1*50, 0.1)
        # self.x[:, 3] = torch.arange(-1.5*5, 1.5*5, 0.015)
        
        ### weights; slope m
        # self.w = torch.tensor([[1.0], [1.0], [1.0]])
        self.w = torch.tensor([ [1.0] for i in range(self.x.shape[1]) ])

        # CONSISTENCY CHECK before PyTorch compute-graph compile-time: do the # x[1] dims = # w[0] dims so we can do matmul?
        assert self.x.shape[1] == self.w.shape[0], f"self.x.shape[1]={self.x.shape[1]} != self.w.shape[0]={self.w.shape[0]}"
        print(f"self.x.shape={self.x.shape}, self.w.shape={self.w.shape}")

        ### bias; intercept b
        self.b = 1
        ### linear function of weights & sample data
        self.func = torch.mm(self.x, self.w) + self.b                   # y = mX + b
        ### target: "actual" y = wX+b + Gaussian noise.
        self.y = self.func + 0.2 * torch.randn((self.x.shape[0], 1))
        # convenience
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

# instantiate dataset obj & generate the dataset it holds.
data_set = Data()

#%%
class MultipleLinearRegression(torch.nn.Module):
    # ctor
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
    
    # Y = estimator(X)
    def forward(self, X):
        y_pred = self.linear(X)  # aka Y_hat
        return y_pred
    
#%%
# instantiate the model
# MLR_model = MultipleLinearRegression(2, 1)
MLR_model = MultipleLinearRegression(data_set.x.shape[1], 1)
# define model-optimization functions & their hyperparams
optimizer = torch.optim.SGD(MLR_model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()
# train_loader = DataLoader(dataset=data_set, batch_size=2)
train_loader = DataLoader(dataset=data_set, batch_size=data_set.x.shape[0]//2)

#%%[markdown]
### Train the model

#%%
losses = []
n_epochs = 20

for epoch in range(n_epochs):

    # dataset mini-batch loop
    p = 0
    for x, y in train_loader:
        # predict on input x
        y_pred = MLR_model(x)
        # calc expected vs actual delta (aka error, aka loss, aka "residual"...)
        loss = criterion(y_pred, y)
        if p < 2:
            # always print first loss per epoch
            if p == 0:
                print(f"[epoch={epoch}]\n\tx={x},\n\ty={y},\n\ty_pred={y_pred}, loss={loss}")
                p += 1
            # print 1 more if loss goes nan
            elif float('nan') == loss:
                print(f"**DETECTED loss=NaN** [epoch={epoch}]\n\tx={x},\n\ty={y},\n\ty_pred={y_pred}, loss={loss}")
                p += 1
        # keep track of error
        losses.append(loss.item())
        # always need to zero-out the optimizer's gradients before updating them
        optimizer.zero_grad()
        # backprop error updates through the weights
        loss.backward()
        # take 1 step along the loss surface
        optimizer.step()

    print(f"epoch={epoch}, loss={loss}")    # loss: versus C & Java, it's surprising that in Python variables created in a loop remain in-scope outside of that loop, within the scope of the enclosing function: https://stackoverflow.com/questions/3611760/scoping-in-python-for-loops
print("Done training.")

#%%[markdown]
### Results
plt.plot(losses)
plt.xlabel("Iterations ")
plt.ylabel("total loss ")
plt.show()
# %%
