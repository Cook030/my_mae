# test_final.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ViT_Classifier, MAE_Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
dummy_enc = MAE_Encoder()
model = ViT_Classifier(dummy_enc, num_classes=10)
model.load_state_dict(torch.load("vit-t-mae-finetuned.pth", map_location="cpu"))
model.eval().to(device)

# 测试集（无增强）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=256, shuffle=False)

correct = total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"\n🎯 Final Independent Test Accuracy: {100 * correct / total:.2f}%\n")