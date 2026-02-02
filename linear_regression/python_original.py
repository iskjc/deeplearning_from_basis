import numpy as np
np.random.seed(0)
x=np.random.rand(100,1)
y=3*x+2+0.1*np.random.randn(100,1)
w=np.random.randn(1)
b=0.0
lr=0.1
epochs=1000
for epoch in range(epochs):
    y_pred=w*x+b
    loss=np.mean((y_pred-y)**2)
    dw=np.mean(2*(y_pred-y)*x)
    db=np.mean((y_pred-y)*2)
    w-=lr*dw
    b-=lr*db
    if epoch%100==0:
        print(f'Epoch {epoch}, Loss:{loss:.4f} w;{w} b:{b} ')
print(f'Final parameters: w:{w}, b:{b}')
