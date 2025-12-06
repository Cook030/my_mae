import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ViT_Classifier, MAE_Encoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练模型
    ckpt = torch.load("vit-t-mae-400ep-ep400.pth", map_location="cpu", weights_only=False)
    if hasattr(ckpt, '_orig_mod'):
        ckpt = ckpt._orig_mod

    # 构建分类模型（patch_size=2）
    dummy_enc = MAE_Encoder()  # 默认 patch_size=2
    model = ViT_Classifier(dummy_enc, num_classes=10)

    # 加载权重
    model.cls_token = ckpt.encoder.cls_token
    model.pos_embedding = ckpt.encoder.pos_embedding
    model.patchify.load_state_dict(ckpt.encoder.patchify.state_dict())
    model.transformer.load_state_dict(ckpt.encoder.transformer.state_dict())
    model.layer_norm.load_state_dict(ckpt.encoder.layer_norm.state_dict())

    # ✅ 关键：全部参数可训练（微调）
    for param in model.parameters():
        param.requires_grad_(True)

    model = model.to(device)

    # 数据增强（微调时推荐使用）
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    val_set   = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # 微调用较小学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_acc = 0.0
    for epoch in range(30):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/30"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

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
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "vit-t-mae-finetuned.pth")

    print(f"🎉 Final Fine-tuned Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()