# visualize_predictions.py
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import ViT_Classifier, MAE_Encoder

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
testset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

model = ViT_Classifier(MAE_Encoder(), num_classes=10)
model.load_state_dict(torch.load("vit-t-mae-finetuned.pth", map_location="cpu"))
model.eval()

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    img, label = testset[i]
    with torch.no_grad():
        pred = model(img.unsqueeze(0)).argmax(1).item()
    ax = axes[i//5, i%5]
    ax.imshow(img.permute(1,2,0) * 0.5 + 0.5)  # 反归一化
    ax.set_title(f"True: {classes[label]}\nPred: {classes[pred]}", color="green" if pred==label else "red")
    ax.axis('off')

plt.tight_layout()
plt.savefig("predictions.png", dpi=150)
print("✅ Saved predictions.png")