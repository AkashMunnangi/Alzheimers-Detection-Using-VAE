import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from model import VAE, LatentClassifier

# ===============================
# CONFIG
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 128
BATCH_SIZE = 32
VAE_EPOCHS = 15
CLS_EPOCHS = 20
LR = 1e-3

os.makedirs("models", exist_ok=True)
os.makedirs("training_results", exist_ok=True)

# ===============================
# DATA TRANSFORMS (FIXED SIZE)
# ===============================
transform = transforms.Compose([
    transforms.Resize((144, 144)),   # ✅ FIXED
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ===============================
# DATASETS
# ===============================
train_dataset = datasets.ImageFolder("train", transform=transform)
test_dataset = datasets.ImageFolder("test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print("Class names:", class_names)

# Save class names
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

# ===============================
# MODELS
# ===============================
vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
classifier = LatentClassifier(latent_dim=LATENT_DIM).to(DEVICE)

# ===============================
# VAE LOSS
# ===============================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# ===============================
# PHASE 1: TRAIN VAE
# ===============================
print("\n===============================")
print("Training VAE")
print("===============================")

vae_optimizer = optim.Adam(vae.parameters(), lr=LR)
vae.train()

for epoch in range(VAE_EPOCHS):
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(DEVICE)

        recon, mu, logvar = vae(images)
        loss = vae_loss(recon, images, mu, logvar)

        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{VAE_EPOCHS}] | Loss: {total_loss:.2f}")

torch.save(vae.state_dict(), "models/vae.pth")
print("✓ VAE saved")

# ===============================
# PHASE 2: TRAIN CLASSIFIER
# ===============================
print("\n===============================")
print("Training Latent Classifier")
print("===============================")

for param in vae.parameters():
    param.requires_grad = False

labels = [label for _, label in train_dataset.samples]
class_counts = torch.bincount(torch.tensor(labels))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=LR)

classifier.train()

for epoch in range(CLS_EPOCHS):
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            mu, _ = vae.encode(images)

        outputs = classifier(mu)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{CLS_EPOCHS}] | Accuracy: {acc:.2f}%")

torch.save(classifier.state_dict(), "models/classifier.pth")
print("✓ Classifier saved")

# ===============================
# EVALUATION
# ===============================
print("\nEvaluating model...")

classifier.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        mu, _ = vae.encode(images)
        outputs = classifier(mu)

        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("F1 Score:", f1)

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("training_results/confusion_matrix.png")
plt.close()

print("✓ Training complete")
