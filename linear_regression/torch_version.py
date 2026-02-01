import torch
torch.manual_seed(0)
x=torch.rand(100,1)
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
y=3*x+2+0.1*torch.randn(100,1)
model=torch.nn.Linear(10,1)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
epochs=1000
for epoch in range(epochs):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%100==0:
        [w,b]=model.parameters()
        print(f'Epoch {epoch}, Loss:{loss.item():.4f} w;{w.item()} b:{b.item()} ')
