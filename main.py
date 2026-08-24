import torch
from tqdm import tqdm
class Dense:
    def __init__(self,input_count,output_count,activation,device,rg):
        self.input_count = input_count
        self.output_count= output_count
        self.activation = activation 
        self.weights = (torch.randn(input_count,output_count,device=device,dtype=D_TYPE)*2/(input_count+output_count)**0.5).requires_grad_(rg)
        self.bias = (torch.zeros(1,output_count,device=device,dtype=D_TYPE)).requires_grad_(rg)

class nn:
    def __init__(self,*layers:list[Dense]):
        self.layers = layers

    def activate(self,Z,activation):
        activation = activation.lower().strip()
        if(activation == "relu"): return torch.relu(Z)
        elif(activation == "softmax"): return torch.softmax(Z,dim=-1)
        else:return Z

    def forward(self,x):
        pred = x
        for i in self.layers:
            pred = pred@i.weights+i.bias
            pred = self.activate(pred,i.activation)
        return pred

class CNN:
    def __init__(self, r, c, device, rg):
        self.conv_Metrix = (torch.randn(r, c, device=device,dtype=D_TYPE) * 0.01).requires_grad_(rg)

    def conv2d(self, arr, stride=1):
        batch_size, H, W = arr.shape
        r, c = self.conv_Metrix.shape
        patches = arr.unfold(1, r, stride).unfold(2, c, stride)
        out_h, out_w = patches.shape[1], patches.shape[2]
        patches = patches.reshape(batch_size, out_h, out_w, -1)
        return (patches * self.conv_Metrix.view(-1)).sum(dim=-1)

    def max_pool(self, arr, r, c, stride=1):
        batch_size, H, W = arr.shape
        patches = arr.unfold(1, r, stride).unfold(2, c, stride)
        out_h, out_w = patches.shape[1], patches.shape[2]
        patches = patches.reshape(batch_size, out_h, out_w, -1)
        return patches.max(dim=-1).values
    
    def relU(self, Z):
        return torch.clamp(Z, min=0)
    
    def forward(self,X,ffn):
        X = self.conv2d(X,1)
        X = self.relU(X)
        X = self.max_pool(X,r=3,c =3,stride=1)
        X = self.conv2d(X,1)
        X = self.relU(X)
        X = self.max_pool(X,r=3,c =3,stride=1)
        X = X.reshape(X.shape[0],-1)
        return ffn.forward(X)

    def backward(self, loss, lr):
        loss.backward()
        with torch.no_grad():
            self.step_count += 1
            for i, p in enumerate(self.all_params):
                if p.grad is None:
                    continue
                self.moment1[i].mul_(0.9).add_(p.grad, alpha=0.1)
                self.moment2[i].mul_(0.999).add_(p.grad * p.grad, alpha=0.001)
                m1_hat = self.moment1[i] / (1 - 0.9 ** self.step_count)
                m2_hat = self.moment2[i] / (1 - 0.999 ** self.step_count)
                update = lr * m1_hat / (torch.sqrt(m2_hat) + 1e-8)
                p.sub_(update)

    def get_params(self):
        return self.all_params

class PDCNN:
    def __init__(self,*layers:list[Dense],**kwargs:list[Dense]):
        self.num_dialation = kwargs["num_dialation"]
        self.r = kwargs['r']
        self.c = kwargs['c']
        self.layers = layers
        self.mlp_layers = kwargs['mlp']

    def forward(self,X):
        l:list[CNN] = []
        f:list[nn] = []
        p:list[torch.Tensor] =[]
        for _ in range(self.num_dialation):
            l.append(CNN(self.r,self.c,device=DEVICE,rg=RG))
            f.append(nn(self.layers))
        for i in range(self.num_dialation):
            p.append(l[i].forward(X,f[i]))
        pred = torch.cat(p,dim=0)
        mlp = nn(self.mlp_layers)
        pred = mlp.forward(pred)
        return pred
        
    def backward(self,X):
        ...
    def one_hot_encode(self, Y: torch.Tensor) -> torch.Tensor:
        labels = Y.reshape(-1)
        encoded = torch.zeros(
            labels.numel(),
            int(labels.max().item()) + 1,
            device=Y.device,
            dtype=torch.get_default_dtype(),
        )
        return encoded.scatter_(1, labels.unsqueeze(1), 1.0)

    def fit(self,X,Y,n_batch = 32,lr = 0.01,epochs =100):
        Y = self.one_hot_encode(Y)
        pbar = tqdm(range(epochs),desc="Training",colour="red")
        for i in pbar:
            order = torch.randperm(X.shape[0], device=X.device)
            epoch_loss = 0.0
            bbar = tqdm(range(0, X.shape[0], n_batch),desc=f"Batch: {i + 1}",leave=False)
            for start in bbar:
                batch_indices = order[start:start + n_batch]
                x_batch = X[batch_indices]
                y_batch = Y[batch_indices]
                pred = self.forward(x_batch)
                loss = torch.nn.functional.cross_entropy(pred, y_batch)
                epoch_loss += loss.item() * x_batch.shape[0]
                bbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.set_postfix(loss=f"{epoch_loss / X.shape[0]:.4f}")


    def predict(self,X_test):
        return torch.argmax(self.forward(X_test))
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    EPOCHS =5
    LR = 0.001
    RG = True
    NUM_DIALATION = 4
    D_TYPE = torch.float64
    torch.manual_seed(200)
    train = torch.load("Training.pt")
    X_train = train["images"]
    y_train = train["labels"]
    model = PDCNN(
        Dense(X_train.shape[0],128,)
    )
    model.fit(X_train,y_train,n_batch=64,lr = 0.01,epochs=100)
    X_train = None
    y_train = None
    test = torch.load("Testing.pt")
    x_test = test["images"]
    y_test = test["labels"]
    y_pred = model.predict(x_test)
    print(torch.mean((y_pred == y_test).to(int)))