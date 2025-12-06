# linear_probe.py
import torch, torchvision, tqdm, argparse, os
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from model import MAE_ViT       # 你的 model.py

def main():
    device = 'cuda'
    # 1. 加载预训练 encoder 并冻结
    ckpt = torch.load('vit-t-mae-400ep.pth', map_location='cpu', weights_only=True)
    encoder = ckpt.encoder if hasattr(ckpt, 'encoder') else ckpt.module.encoder
    encoder.eval().requires_grad_(False).to(device)

    # 2. 分类头
    cls_head = torch.nn.Linear(192, 10).to(device)

    # 3. 数据
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_set = torchvision.datasets.CIFAR10('data', train=True,  transform=transform)
    val_set   = torchvision.datasets.CIFAR10('data', train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True,  num_workers=12, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=512, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True)

    # 4. 优化器 & 调度器
    optim = torch.optim.SGD(cls_head.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 5. 训练 100 epoch
    for epoch in range(100):
        cls_head.train()
        for x, y in tqdm.tqdm(train_loader, leave=False, ncols=80):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.no_grad():
                out = encoder(x)                      # [B, 197, 192]
                feat = out[0][:, 1:].mean(dim=1)      # 去 cls token 后平均
            logits = cls_head(feat)
            loss = loss_fn(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
        scheduler.step()

        # 6. 验证
        cls_head.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device)
                feat = encoder(x)[0][:, 1:].mean(dim=1)
                logits = cls_head(feat)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f'Epoch {epoch+1}/100  val_acc={acc:.2%}')

    torch.save(cls_head, 'linear_probe_head.pth')
    print('Linear probing done, final acc =', acc)

if __name__ == '__main__':
    main()