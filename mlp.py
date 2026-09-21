import torch

class Dense:
    def __init__(self, input_count: int, output_count: int, activation: str):
        self.weights = torch.randn(input_count, output_count) * 2/(input_count)**0.5  # scaled for better training
        self.bias = torch.zeros(1, output_count)
        self.activation = activation.lower().strip()

class MLP:
    def __init__(self, *args: Dense):
        self.layers = args
        self.result = []

    def activate(self, Z, activation: str):
        if activation == "relu":
            return torch.relu(Z)
        elif activation == "sigmoid":
            return torch.sigmoid(Z)
        else:
            return Z
    
    def forward(self, X):
        pred = X
        self.result = [X]
        for layer in self.layers:
            pred = pred @ layer.weights + layer.bias
            pred = self.activate(pred, activation=layer.activation)
            self.result.append(pred)
        return pred

    def deactivate(self, A, activation):
        if activation == "relu":
            return (A > 0).float()
        elif activation == "sigmoid":
            return A * (1 - A)
        else:
            return torch.ones_like(A)

    def backward(self, y, pred, lr):
        dA = pred - y
        dZ = None
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = self.result[i]
            A_curr = self.result[i + 1] 
            d_act = self.deactivate(A_curr, layer.activation)
            if i == len(self.layers) - 1:
                dZ = dA * d_act
            else:
                dZ = (dZ @ next_weights.T) * d_act
            dW = A_prev.T @ dZ
            dB = torch.sum(dZ, dim=0, keepdim=True)
            next_weights = layer.weights.clone()
            layer.weights -= lr * dW
            layer.bias -= lr * dB

    def predict(self, x):
        preds = self.forward(x)
        return (preds >= 0.5).int()

    def fit(self, x, y, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            A = self.forward(x)
            loss = torch.mean((A - y) ** 2)
            self.backward(y, A, lr)
            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

if __name__ == "__main__":
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float)
    torch.manual_seed(20)
    model = MLP(
        Dense(2, 10, "relu"),
        Dense(10,40,"relu"),
        Dense(40,4,""),
        Dense(4, 1, "sigmoid")
    )
    
    model.fit(x, y, epochs=1000, lr=0.1)
    prd = model.predict(x)
    print("Predictions:\n", prd)
    print("Acc: ", torch.mean((prd == y).float()).item())