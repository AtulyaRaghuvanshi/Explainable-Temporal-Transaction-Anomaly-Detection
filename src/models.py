import numpy as np,torch,joblib
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
class Autoencoder(nn.Module):
    def __init__(self,d,h1=256,h2=128,z=32):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(d,h1),nn.ReLU(),nn.Linear(h1,h2),nn.ReLU(),nn.Linear(h2,z));self.decoder=nn.Sequential(nn.Linear(z,h2),nn.ReLU(),nn.Linear(h2,h1),nn.ReLU(),nn.Linear(h1,d))
    def forward(self,x):return self.decoder(self.encoder(x))
def train_ae(X,epochs,batch,lr,h1,h2,z,max_rows):
    X=np.asarray(X,np.float32);sc=StandardScaler();Xs=sc.fit_transform(X);rng=np.random.default_rng(42)
    if len(Xs)>max_rows:Xs=Xs[rng.choice(len(Xs),max_rows,replace=False)]
    dev='cuda' if torch.cuda.is_available() else 'cpu';m=Autoencoder(X.shape[1],h1,h2,z).to(dev);dl=DataLoader(TensorDataset(torch.tensor(Xs,dtype=torch.float32)),batch_size=batch,shuffle=True);opt=torch.optim.Adam(m.parameters(),lr=lr);lossf=nn.MSELoss()
    for ep in range(epochs):
        total=0
        for (xb,) in dl:
            xb=xb.to(dev);opt.zero_grad();loss=lossf(m(xb),xb);loss.backward();opt.step();total+=loss.item()*len(xb)
        print(f'AE epoch {ep+1}/{epochs}: {total/len(dl.dataset):.6f}')
    return m,sc,dev
def reconstruction_error(m,sc,X,dev):
    Xs=sc.transform(np.asarray(X,np.float32));m.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(Xs),8192):
            xb=torch.tensor(Xs[i:i+8192],dtype=torch.float32).to(dev);out.append(((Xs[i:i+8192]-m(xb).cpu().numpy())**2).mean(axis=1))
    return np.concatenate(out)
def save_ae(m,sc,path):joblib.dump({'state_dict':m.state_dict(),'input_dim':m.encoder[0].in_features,'scaler':sc},path)
