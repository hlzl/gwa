import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import timm
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from scipy.stats import kurtosis


class Tap(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.latent_embeddings = None

    def forward(self, x):
        self.latent_embeddings = x.detach()
        return self.head(x)


def compute_alignment(head_layer, logits, target):
    if head_layer.latent_embeddings is None:
        raise RuntimeError("Tap did not capture any output.")

    # Closed-form gradients for alignment do not need autograd
    with torch.no_grad():
        diff = (
            nn.functional.softmax(logits, dim=-1)
            - torch.nn.functional.one_hot(target, num_classes=logits.size(-1)).float()
        )

        # Numerator: - x_b^T (W^T diff_b) for each sample b
        W = head_layer.head.weight.detach()  # C x D
        v = diff @ W  # B x D
        dot_product = -(head_layer.latent_embeddings * v).sum(dim=-1)  # B

        # Denominator: ||grad|| * ||W||
        grad_norm = head_layer.latent_embeddings.norm(dim=-1) * diff.norm(dim=-1)  # B
        weight_norm = W.norm()  # scalar

        alignment_scores = dot_product / (grad_norm * weight_norm + 1e-12)

    return alignment_scores


# 1. Load a ConvNeXt model and CIFAR‑10 dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/path/to"  # NOTE
n_classes = 10
epochs = 50
image_size = 32

# Create model and adapt final layer for CIFAR‑10
model = timm.create_model("convnextv2_pico", kernel_sizes=3, patch_size=1)
# NOTE: layer needs to be adapted based on the model used
model.head.fc = nn.Linear(model.head.fc.in_features, n_classes)
model.head.fc = Tap(model.head.fc)
head_layer = model.head.fc  # set here for alignment computation later
model = model.to(device)

train_transform = T.Compose(
    [
        T.Resize(image_size + 4),
        T.RandAugment(1),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
eval_transform = T.Compose(
    [
        T.RandomCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root=dataset_path, train=True, download=True, transform=train_transform
)
train_set, val_set = torch.utils.data.random_split(
    train_set,
    [int(0.9 * len(train_set)), int(0.1 * len(train_set))],
    generator=torch.Generator().manual_seed(1),
)
test_set = torchvision.datasets.CIFAR10(
    root=dataset_path, train=False, download=True, transform=eval_transform
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False)

optim = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.0)
criterion = torch.nn.CrossEntropyLoss()


# 2. Training and Evaluation
gwa = []
val_acc = []
test_acc = []
os.makedirs("logs", exist_ok=True)
for epoch in range(epochs):
    # Training
    model.train()
    total_train_loss = 0
    dataset_alignment = []
    for images, targets, *_ in tqdm(
        train_loader, desc=f"Epoch [{epoch+1}/{epochs}] - Train", leave=False
    ):
        optim.zero_grad()
        # Forward pass and loss on the autograd graph
        logits = model(images.cuda())
        loss = criterion(logits, targets.cuda())
        loss.backward()
        optim.step()

        total_train_loss += loss.item()
        alignment = compute_alignment(head_layer, logits, targets.cuda())  # NOTE
        dataset_alignment.append(alignment)
    dataset_alignment = torch.concat(dataset_alignment).cpu().numpy()
    gwa.append(np.mean(dataset_alignment, axis=-1) / (kurtosis(dataset_alignment, axis=-1) + 1.2))

    # NOTE: optional - create plots of alignment distribution
    if epoch in [0, 1, 2, 9, 19, 49, 69, 99]:
        plt.hist(dataset_alignment, bins=100, label="Alignment Scores")
        plt.legend()
        plt.title("Aligment Scores")
        plt.savefig(f"logs/torch_alignment_dist_epoch{epoch}.png")
        plt.clf()

    print(
        f"Epoch [{epoch+1}/{epochs}]\n"
        f"Train Loss: {total_train_loss/len(train_loader):.4f}, "
        f"GWA: {gwa[epoch]:.4f}"
    )

    # Validation and Test
    model.eval()
    with torch.no_grad():
        for val_test, loader in zip(["Val", "Test"], [val_loader, test_loader]):
            total_loss = 0
            correct = 0
            for images, targets, *_ in tqdm(
                loader, desc=f"Epoch [{epoch+1}/{epochs}] - {val_test}", leave=False
            ):
                output = model(images.cuda())
                loss = criterion(output, targets.cuda())
                total_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == targets.cuda()).sum().item()

            if val_test == "Val":
                val_acc.append(100 * correct / len(val_set))
                print(
                    f"Val Loss: {total_loss/len(loader):.4f}, "
                    f"Val Accuracy: {val_acc[epoch]:.2f}% "
                )
            else:
                test_acc.append(100 * correct / len(test_set))
                print(
                    f"Test Loss: {total_loss/len(loader):.4f}, "
                    f"Test Accuracy: {test_acc[epoch]:.2f}% "
                )

    joblib.dump(
        {
            "val_acc": val_acc[epoch],
            "test_acc": test_acc[epoch],
            "gwa": gwa[epoch],
            "cosine": dataset_alignment,
        },
        f"logs/torch_metrics_{epoch}.joblib",
    )


# 3. Normalized Values Over Time
val_acc = np.array(val_acc)
val_acc = (val_acc - val_acc.min()) / abs(val_acc - val_acc.min()).max()
plt.plot(val_acc, label="Val Acc")
gwa = np.array(gwa)
gwa = (gwa - gwa.min()) / abs(gwa - gwa.min()).max()
plt.plot(gwa, label="GWA")
plt.legend()
plt.savefig("logs/torch_acc-vs-gwa.png")
