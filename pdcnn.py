import torch
import torch.nn as nn
from tqdm import tqdm
import os
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class Dense:
    def __init__(self , input_size :int , output_size :int ,activation :str) -> None:
        scale = (2.0/input_size)**0.5
        self.weights    = (torch.randn(input_size,output_size,device=DEVICE)*scale).requires_grad_(True)
        self.bias       = torch.randn(1,output_size,device=DEVICE).requires_grad_(True)
        self.activation = activation.strip().lower()

    def get_params(self) -> list[torch.Tensor]:
        return [self.weights,self.bias]

class NN:
    def __init__(self , *layers :list[Dense]) -> None:
        self.layers = layers

    def _activate(self, Z : torch.Tensor , activation :str) -> torch.Tensor:
        if   activation == "relu":
            return torch.maximum(torch.tensor(0.0, device=Z.device), Z)
        elif activation == "softmax":
            return torch.softmax(Z,dim=-1)
        else :
            return Z

    def forward(self, X: torch.Tensor ) -> torch.Tensor:
        pred = X
        for layer in self.layers:
            Z = pred @ layer.weights + layer.bias
            pred = self._activate(Z,layer.activation)
        return pred

    def get_params(self) -> list[torch.Tensor]:
        p = []
        for layer in self.layers:
            p.extend(layer.get_params())
        return p

class CNN(nn.Module):
    def __init__(
            self, kernal_size :int, in_channels :int,
            out_channels :int,stride :int, padding :int,
            n_times :int, dialation :int
        ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_times):
            c_in = in_channels if i==0 else out_channels
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels =c_in,
                    out_channels=out_channels,
                    kernel_size =kernal_size,
                    stride      =stride,
                    padding     =padding,
                    dilation    =dialation,
                    device=DEVICE
                ),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride     =2,
                )
            )
            self.blocks.append(block)
    
    def forward(self, X :torch.Tensor) -> torch.Tensor:
        pred = X
        for block in self.blocks:
            pred = block(pred)
        return torch.flatten(pred,start_dim=1)


class PDCNN:
    def __init__(
            self,n_dialation :int, kernal_size :int,
            in_channels :int,out_channels:int,stride :int,
            n_time_conv :int, MLP: NN,ffn : list[NN]
        ) -> None:
        self.n_dialation = n_dialation
        self.ffn = ffn
        self.cnn_layers = nn.ModuleList()
        self.MLP = MLP
        for dialation in range(1,n_dialation+1):
            padding = ((kernal_size - 1) * dialation) // 2
            self.cnn_layers.append(
                CNN(
                    kernal_size =kernal_size,
                    in_channels =in_channels,
                    out_channels=out_channels,
                    stride   = stride,
                    n_times  = n_time_conv,
                    dialation= dialation,
                    padding  = padding
                )
            )
    def forward(self, X :torch.Tensor) -> torch.Tensor:
        pred = []
        for i in range(self.n_dialation):
            pred.append(self.ffn[i].forward(self.cnn_layers[i].forward(X)))
        pred = torch.cat(pred,dim = -1)
        return self.MLP.forward(pred)

    def get_params(self) -> list[torch.Tensor]:
        p = list(self.cnn_layers.parameters())
        for f in self.ffn:
            p.extend(f.get_params())
        p.extend(self.MLP.get_params())
        return p
    
    def backward(self,loss: torch.Tensor,lr :float,params: list[torch.Tensor]) -> None:
        loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.data -= lr * p.grad.data
        

    def fit(self, x: torch.Tensor, y: torch.Tensor, n_batchs: int = 32, lr: float = 0.01, epochs: int = 100) -> None:
        criterion = nn.CrossEntropyLoss()
        n_samples = x.shape[0]
        t_bar = tqdm(range(epochs),desc="Training",colour="green")
        for _ in t_bar:
            epoch_loss = 0.0
            indices = torch.randperm(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            b_bar = tqdm(range(0, n_samples, n_batchs),desc="Batch: 0",colour="red")
            for i in b_bar:
                x_batch = x_shuffled[i:i + n_batchs].to(DEVICE)
                y_batch = y_shuffled[i:i + n_batchs].to(DEVICE)
                params = self.get_params()
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

                preds = self.forward(x_batch)
                loss = criterion(preds, y_batch)
                self.backward(loss,lr,params)
                epoch_loss += loss.item() * x_batch.shape[0]
                b_bar.set_description(f"Batch: {i+1}")
                b_bar.set_postfix(Loss = f"{loss.item():.4f}")
            epoch_loss /= n_samples
            t_bar.set_postfix(Loss = f"{epoch_loss:.4f}")

    def predict(self,X :torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)

if __name__ == '__main__':
    MODEL_NUMBER = int(open("model.inf","r").read())
    SAVE_DIR     = r"MODELS/"
    BATCH_SIZE   = 50
    IMG_SIZE     = 224
    OUT_CHANNELS = 32
    NUM_CLASSES  = 4
    EPOCHS       = 1
    IN_CHANNELS  = 3
    KERNAL_SIZE  = 3
    N_TIME_CONV  = 3
    N_DIALATIONS = 5
    LR           = 0.01
    
    spatial_dim = IMG_SIZE // (2 ** N_TIME_CONV)
    cnn_flattened_dim = OUT_CHANNELS * spatial_dim * spatial_dim 
    ffn      = []
    branch_out_dim = 128
    for i in range(N_DIALATIONS):
        ffn.append(
            NN(
                Dense(cnn_flattened_dim, branch_out_dim, "relu"),
                Dense(branch_out_dim,branch_out_dim,"relu")
            )
        )
    mlp_input_dim = branch_out_dim * N_DIALATIONS
    mlp = NN(Dense(mlp_input_dim, NUM_CLASSES, "none"))

    model = PDCNN(
        n_dialation = N_DIALATIONS,
        kernal_size = KERNAL_SIZE,
        in_channels = IN_CHANNELS,
        out_channels= OUT_CHANNELS,
        stride      = 1,
        n_time_conv = 3,
        MLP         = mlp,
        ffn         = ffn
    )
    train = torch.load("Training.pt",weights_only=True)
    x_train = train["images"]
    y_train = train["labels"]
    del train
    model.fit(
        x = x_train,
        y = y_train,
        n_batchs = BATCH_SIZE,
        lr       = LR,
        epochs   = EPOCHS
    )
    del x_train, y_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test    = torch.load("Testing.pt",weights_only=True)
    x_test  = test["images"].to(DEVICE)
    y_test  = test["labels"].to(DEVICE)
    del test
    ypred = model.predict(x_test)
    accuracy = (ypred == y_test).float().mean() * 100
    print(f"Accuracy: {accuracy.item():.2f}%")
    with open("model.inf", "w") as f:
        f.write(str(MODEL_NUMBER + 1))
    params = model.get_params()
    save_path = os.path.join(SAVE_DIR, f"MODEL_{MODEL_NUMBER}_AC_{str(accuracy.item())[:5]}.pt")
    params = model.get_params()
    torch.save(params, save_path)