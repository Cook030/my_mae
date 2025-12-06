# import os
# import argparse
# import math
# import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor, Compose, Normalize
# from tqdm import tqdm

# from model import *
# from utils import setup_seed

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('-bs','--batch_size', type=int, default=4096)
#     parser.add_argument('--max_device_batch_size', type=int, default=128)
#     parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
#     parser.add_argument('--weight_decay', type=float, default=0.05)
#     parser.add_argument('--mask_ratio', type=float, default=0.75)
#     parser.add_argument('--total_epoch', type=int, default=2000)
#     parser.add_argument('--warmup_epoch', type=int, default=200)
#     parser.add_argument('--model_path', type=str, default='vit-t-mae.pth')

#     args = parser.parse_args()

#     setup_seed(args.seed)

#     batch_size = args.batch_size
#     load_batch_size = min(args.max_device_batch_size, batch_size)

#     assert batch_size % load_batch_size == 0
#     steps_per_update = batch_size // load_batch_size

#     train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
#     val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
#     dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
#     writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
#     if device == 'cuda':
#         net = torch.nn.DataParallel(model)

#     optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
#     lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

#     step_count = 0
#     optim.zero_grad()
#     for e in range(args.total_epoch):
#         model.train()
#         losses = []
#         train_step = len(dataloader)
#         with tqdm(total=train_step,desc=f'Epoch {e+1}/{args.total_epoch}',postfix=dict,mininterval=0.3) as pbar:
#             for img, label in iter(dataloader):
#                 step_count += 1
#                 img = img.to(device)
#                 predicted_img, mask = model(img)
#                 loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
#                 loss.backward()
#                 if step_count % steps_per_update == 0:
#                     optim.step()
#                     optim.zero_grad()
#                 losses.append(loss.item())
#                 pbar.set_postfix(**{'Loss' : np.mean(losses)})
#                 pbar.update(1)
#         lr_scheduler.step()
#         avg_loss = sum(losses) / len(losses)
#         writer.add_scalar('mae_loss', avg_loss, global_step=e)
#         # print(f'In epoch {e}, average traning loss is {avg_loss}.')

#         ''' visualize the first 16 predicted images on val dataset'''
#         model.eval()
#         with torch.no_grad():
#             val_img = torch.stack([val_dataset[i][0] for i in range(16)])
#             val_img = val_img.to(device)
#             predicted_val_img, mask = model(val_img)
#             predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
#             img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
#             img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
#             writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
#         ''' save model '''
#         torch.save(model, args.model_path)


import os
import math
import torch, torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler   # FP16
import tqdm
from einops import rearrange
import numpy as np

from model import MAE_ViT
from utils import setup_seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-bs', '--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)  # 4090D 24G 可吃满
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=400)   # 收敛即够用
    parser.add_argument('--warmup_epoch', type=int, default=40)  # 按比例缩小
    parser.add_argument('--model_path', type=str, default='vit-t-mae-400ep.pth')
    return parser.parse_args()

def main():
    args = get_args()
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 数据：CIFAR10 自动下载 ----
    transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
    train_set = torchvision.datasets.CIFAR10('data', train=True,  download=True, transform=transform)
    val_set   = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    loader = DataLoader(train_set,
                        batch_size=args.max_device_batch_size,
                        shuffle=True,
                        num_workers=12,
                        pin_memory=True,
                        persistent_workers=True)

    # ---- 模型 + compile ----
    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    if hasattr(torch, 'compile'):          # PyTorch2.x
        model = torch.compile(model)
    model = nn.DataParallel(model) if device == 'cuda' else model

    # ---- 优化器 & 调度器 ----
    steps_per_update = args.batch_size // args.max_device_batch_size
    optim = torch.optim.AdamW(model.parameters(),
                              lr=args.base_learning_rate * args.batch_size / 256,
                              betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch+1)/(args.warmup_epoch+1e-8),
                                0.5*(math.cos(epoch/args.total_epoch*math.pi)+1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    # ---- 混合精度 ----
    scaler = GradScaler()

    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    step_cnt = 0
    optim.zero_grad()

    for epoch in range(args.total_epoch):
        model.train()
        losses = []
        pbar = tqdm.tqdm(loader, total=len(loader), ncols=80,
                         desc=f'Epoch {epoch+1}/{args.total_epoch}')
        for img, _ in pbar:
            img = img.to(device, non_blocking=True)
            with autocast():
                pred, mask = model(img)
                loss = torch.mean((pred - img)**2 * mask) / args.mask_ratio

            scaler.scale(loss).backward()
            step_cnt += 1
            if step_cnt % steps_per_update == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        scheduler.step()
        avg_loss = np.mean(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=epoch)

        # ---- 每 10 epoch 可视化一次 ----
        if epoch % 10 == 0 or epoch == args.total_epoch - 1:
            model.eval()
            with torch.no_grad():
                val_img = torch.stack([val_set[i][0] for i in range(16)]).to(device)
                pred, mask = model(val_img)
                pred = pred * mask + val_img * (1 - mask)
                grid = torch.cat([val_img*(1-mask), pred, val_img], dim=0)
                grid = rearrange(grid, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
                writer.add_image('mae_image', (grid+1)/2, global_step=epoch)

        # ---- 每 50 epoch 存一次 ----
        if (epoch+1) % 50 == 0 or epoch == args.total_epoch - 1:
            torch.save(model.module if hasattr(model, 'module') else model,
                       args.model_path.replace('.pth', f'-ep{epoch+1}.pth'))

    writer.close()
    print('Pre-training done! Checkpoint ->', args.model_path)

if __name__ == '__main__':
    import argparse
    main()