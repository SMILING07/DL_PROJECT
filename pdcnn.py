"""
PDCNN - Parallel Dilated CNN for Brain Tumor Classification
Architecture:
    Stem:  Shared convolutional layers that learn common MRI features
    Branches: 3 parallel dilated CNN branches (in our case we are using dilation 1, 2, 3) for capturing
              local, medium, and large receptive field patterns
    Classifier: Fully connected head with dropout regularization
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
'''
NOTE: Guys use the training only in cuda do the training only 
if cuda is present else dont cause it's gonna take forever to train.
'''
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Convolution Block
class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DilatedBranch(nn.Module):
    """
    A single dilated CNN branch.
    Progressively increases channels while applying the same dilation rate
    across all layers, then uses Global Average Pooling to produce a
    fixed-length feature vector regardless of spatial dimensions.
    """
    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        dilations: list[int],
    ):
        super().__init__()
        assert len(channels) == len(dilations), \
            "channels and dilations must have the same length"

        layers = []
        ch_in = in_channels
        for ch_out, dilation in zip(channels, dilations):
            layers.append(ConvBlock(ch_in, ch_out, dilation=dilation))
            layers.append(nn.MaxPool2d(2))
            ch_in = ch_out

        # Final conv block (no pooling) before GAP
        layers.append(ConvBlock(ch_in, channels[-1], dilation=dilations[-1]))

        self.features = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

# Main model
class PDCNN(nn.Module):
    """
    Parallel Dilated Convolutional Neural Network.
    Input:  (B, 3, 224, 224)  — RGB image
    Output: (B, num_classes)   — raw logits
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Stem: shared feature extractor 
        self.stem = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),  # 224 -> 112
        )
        # Parallel dilated branches 
        self.branches = nn.ModuleList([
            DilatedBranch(64, [128, 256, 512], [1, 1, 1]),  # local features
            DilatedBranch(64, [128, 256, 512], [2, 2, 2]),  # medium features
            DilatedBranch(64, [128, 256, 512], [3, 3, 3]),  # large features
        ])
        concat_dim = sum(b.out_channels for b in self.branches)  # 1536
        # Classifier head 
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self._init_weights()
    # weight initialisation
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    # forward 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        return self.classifier(x)
    
    # predict
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x = x.to(DEVICE, non_blocking=True)
        return torch.argmax(self.forward(x), dim=1)
    
    # evaluation
    @torch.no_grad()
    def _evaluate(self, x, y, batch_size=128):
        self.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, n = 0.0, 0, x.shape[0]
        use_amp = DEVICE == "cuda"
        for i in range(0, n, batch_size):
            xb = x[i : i + batch_size].to(DEVICE, non_blocking=True)
            yb = y[i : i + batch_size].to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self(xb)
                total_loss += criterion(logits, yb).item() * xb.shape[0]
                correct += (logits.argmax(1) == yb).sum().item()
            del xb, yb
        return total_loss / n, correct / n

    # training 
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        batch_size: int = 32,
        lr: float = 5e-4,
        epochs: int = 50,
        checkpoint_dir: str = "checkpoint",
    ):
        self.to(DEVICE)
        self.train()
        os.makedirs(checkpoint_dir, exist_ok=True)

        # optimiser, scheduler, loss
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        steps_per_epoch = math.ceil(x.shape[0] / batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.1,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        use_amp = DEVICE == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        n = x.shape[0]
        best_val_acc = 0.0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        # Training loop 
        for epoch in tqdm(range(epochs), desc="Training", colour="green"):
            self.train()
            epoch_loss = 0.0
            # shuffle indices (on cpu)
            indices = torch.randperm(n)
            # batch loop
            pbar = tqdm(range(0, n, batch_size), leave=False, colour="red")
            for i in pbar:
                # move only this batch to GPU
                xb = x[indices[i : i + batch_size]].to(DEVICE, non_blocking=True)
                yb = y[indices[i : i + batch_size]].to(DEVICE, non_blocking=True)
                xb = self._augment(xb)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(self(xb), yb)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                epoch_loss += loss.item() * xb.shape[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                # free batch from GPU
                del xb, yb

            # end of epoch validation and saving checkpoint
            avg_train_loss = epoch_loss / n
            history["train_loss"].append(avg_train_loss)
            msg = f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}"

            if x_val is not None and y_val is not None:
                val_loss, val_acc = self._evaluate(x_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
                # only saving the current epoch shows good validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(
                        checkpoint_dir, epoch + 1,
                        avg_train_loss, val_loss, val_acc, tag="best",
                    )

            tqdm.write(msg)

        # save final checkpoint
        last_val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
        last_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
        self._save_checkpoint(
            checkpoint_dir, epochs,
            avg_train_loss, last_val_loss, last_val_acc, tag="final",
        )
        return history

    # checkpointing 
    def _save_checkpoint(self, directory, epoch, train_loss, val_loss, val_acc, tag=""):
        state_dict = self._orig_mod.state_dict() if hasattr(self, "_orig_mod") else self.state_dict()
        filename = (
            f"PDCNN_{tag}_ep{epoch}"
            f"_trloss{train_loss:.4f}"
            f"_vlloss{val_loss:.4f}"
            f"_vacc{val_acc * 100:.2f}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            os.path.join(directory, filename),
        )
        print(f"  -> Saved checkpoint: {filename}")

    # data augmentation

    @torch.no_grad()
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        '''random horizontal flip,rotating,increasing or decreasing brightness,
            creating more data variations so that the model does not depend on how it looks the Image
        '''
        # flipping the images in random
        flip = torch.rand(B, 1, 1, 1, device=x.device) > 0.5
        x = torch.where(flip, torch.flip(x, [-1]), x)
        # random rotation ±10°
        angles = (torch.rand(B, device=x.device) - 0.5) * 20 * math.pi / 180
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # random brightness jitter ±10%
        x = x * (1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.2)
        return x.clamp(0, 1)
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # hyperparameters
    '''
    Tune these values as you wish,
    batch_size should be selected based on your gpu vram
    if 8gb vram: use 32
    if 6gb vram: use 16 or 24
    if 4gb vram: use 8
    epochs and learning_rate can be tuned as your wish
    NOTE: DO NOT CHANGE NUM_CLASSES as it contains the number of classes in our dataset
    ''' 
    BATCH_SIZE = 32
    NUM_CLASSES = 4
    EPOCHS = 50
    LR = 5e-4
    # model 
    model = PDCNN(num_classes=NUM_CLASSES)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # data 
    train = torch.load("Training.pt", weights_only=True, map_location="cpu")
    x_train, y_train = train["images"], train["labels"]
    del train

    test = torch.load("Testing.pt", weights_only=True, map_location="cpu")
    x_test, y_test = test["images"], test["labels"]
    del test

    # train
    history = model.fit(
        x=x_train, y=y_train,
        x_val=x_test, y_val=y_test,
        batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS,
    )

    # final evaluation
    del x_train, y_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_pred = model.predict(x_test)
    accuracy = (y_pred == y_test).float().mean() * 100
    print(f"Final Test Accuracy: {accuracy.item():.2f}%")
