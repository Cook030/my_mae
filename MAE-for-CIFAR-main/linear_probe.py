import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MAE_Encoder, ViT_Classifier  # 使用官方分类头

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载完整模型
    ckpt = torch.load("vit-t-mae-400ep-ep400.pth", map_location="cpu")
    if hasattr(ckpt, '_orig_mod'):
        ckpt = ckpt._orig_mod

    # 初始化分类器（结构）
    dummy_enc = MAE_Encoder(mask_ratio=0.75)
    model = ViT_Classifier(dummy_enc, num_classes=10)

    # 加载 encoder 权重
    model.cls_token = ckpt.encoder.cls_token
    model.pos_embedding = ckpt.encoder.pos_embedding
    model.patchify.load_state_dict(ckpt.encoder.patchify.state_dict())
    model.transformer.load_state_dict(ckpt.encoder.transformer.state_dict())
    model.layer_norm.load_state_dict(ckpt.encoder.layer_norm.state_dict())

    # 冻结 backbone
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad_(False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)

    transform = Compose([ToTensor(), Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    val_set   = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(20):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"=> Epoch {epoch+1}  Val Acc: {acc:.2f}%")

    print(f"✅ Final Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()