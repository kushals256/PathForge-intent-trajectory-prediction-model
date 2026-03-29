import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import NuScenesTrajectoryDataset
from model.trajectory_predictor import TrajectoryPredictor
from utils.metrics import compute_min_ade_fde
from tqdm import tqdm

def variety_loss(trajs, K, min_dist=0.5):
    endpoints = trajs[:, :, -1, :]
    B = endpoints.size(0)
    total_penalty = 0
    count = 0
    for i in range(K):
        for j in range(i+1, K):
            dist = torch.norm(endpoints[:, i] - endpoints[:, j], dim=-1)
            penalty = torch.clamp(min_dist - dist, min=0.0)
            total_penalty += penalty.mean()
            count += 1
    return total_penalty / max(count, 1)

def train_epoch(model, loader, optimizer, lambda_conf=0.5, lambda_div=0.5):
    model.train()
    total_loss, total_ade, total_fde = 0, 0, 0

    criterion_reg = nn.SmoothL1Loss(reduction='none')
    criterion_cls = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc='Train')
    for batch in pbar:
        hist = batch['hist'].float()
        gt_fut = batch['fut'].float()
        social = batch['social'].float()
        social_mask = batch['social_mask'].bool()

        optimizer.zero_grad()
        trajs, confs = model(hist, social, social_mask)
        B, K, fut_len, _ = trajs.size()

        gt_expand = gt_fut.unsqueeze(1).expand(B, K, fut_len, 2)
        l2_dists = torch.norm(trajs - gt_expand, dim=-1).mean(dim=-1)
        best_mode_idx = l2_dists.argmin(dim=1)

        batch_indices = torch.arange(B)
        best_trajs = trajs[batch_indices, best_mode_idx]

        loss_reg = criterion_reg(best_trajs, gt_fut).mean()
        loss_cls = criterion_cls(confs, best_mode_idx)
        loss_div = variety_loss(trajs, K)

        loss = loss_reg + lambda_conf * loss_cls + lambda_div * loss_div

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            m_ade, m_fde, _ = compute_min_ade_fde(trajs, gt_fut, K)

        total_loss += loss.item()
        total_ade += m_ade.mean().item()
        total_fde += m_fde.mean().item()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'minADE': f'{m_ade.mean().item():.4f}'})

    n = len(loader)
    return total_loss/n, total_ade/n, total_fde/n

def val_epoch(model, loader):
    model.eval()
    total_ade, total_fde = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val'):
            hist = batch['hist'].float()
            gt_fut = batch['fut'].float()
            social = batch['social'].float()
            social_mask = batch['social_mask'].bool()
            trajs, confs = model(hist, social, social_mask)
            m_ade, m_fde, _ = compute_min_ade_fde(trajs, gt_fut, K=3)
            total_ade += m_ade.mean().item()
            total_fde += m_fde.mean().item()
    n = len(loader)
    return total_ade/n, total_fde/n

if __name__ == '__main__':
    batch_size = 64
    epochs = 200
    lr = 1e-3
    hidden_dim = 128
    warmup_epochs = 5     # FIX #8: LR warmup
    patience = 30         # FIX #7: Early stopping

    print("Loading datasets...")
    train_dataset = NuScenesTrajectoryDataset('v1.0-mini/v1.0-mini', split='train', augment=True)
    val_dataset = NuScenesTrajectoryDataset('v1.0-mini/v1.0-mini', split='val', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data loaded: {len(train_dataset)} train seqs, {len(val_dataset)} val seqs.")

    model = TrajectoryPredictor(input_dim=7, hidden_dim=hidden_dim, K=3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # FIX #8: Warmup + Cosine schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        else:
            # cosine decay after warmup
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_ade = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        t_loss, t_ade, t_fde = train_epoch(model, train_loader, optimizer)
        v_ade, v_fde = val_epoch(model, val_loader)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train - Loss: {t_loss:.4f}, minADE3: {t_ade:.4f}, minFDE3: {t_fde:.4f}")
        print(f"Val   - minADE3: {v_ade:.4f}, minFDE3: {v_fde:.4f}, LR: {current_lr:.6f}")

        if v_ade < best_ade:
            best_ade = v_ade
            epochs_without_improvement = 0
            print(f"🥇 New best validation ADE! Saving model to best_model.pth")
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"⏹️  Early stopping triggered after {patience} epochs without improvement.")
                break
