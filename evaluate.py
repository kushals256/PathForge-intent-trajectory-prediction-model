import torch
import matplotlib.pyplot as plt
import numpy as np
from data.dataset import NuScenesTrajectoryDataset
from model.trajectory_predictor import TrajectoryPredictor
from utils.metrics import compute_min_ade_fde
from torch.utils.data import DataLoader
from tqdm import tqdm

def visualize_prediction(model, dataset, idx=0):
    model.eval()
    batch = dataset[idx]
    hist = batch['hist'].float().unsqueeze(0)
    gt_fut = batch['fut'].float().unsqueeze(0)
    social = batch['social'].float().unsqueeze(0)
    social_mask = batch['social_mask'].bool().unsqueeze(0)

    with torch.no_grad():
        trajs, confs = model(hist, social, social_mask)

    confs = torch.softmax(confs[0], dim=0).numpy()
    trajs = trajs[0].numpy()
    hist = hist[0].numpy()
    gt_fut = gt_fut[0].numpy()
    social = social[0].numpy()
    social_mask = social_mask[0].numpy()

    plt.figure(figsize=(10, 10))
    plt.plot(hist[:, 0], hist[:, 1], 'bo-', label='Past Trajectory', markersize=6, linewidth=2)
    plt.plot(0, 0, 'kx', markersize=12, label='Current Pos')

    gt_plot = np.vstack([[0,0], gt_fut])
    plt.plot(gt_plot[:, 0], gt_plot[:, 1], 'go--', label='Ground Truth Future', markersize=6, linewidth=2)

    colors = ['red', 'orange', 'purple']
    for k in range(3):
        pred_plot = np.vstack([[0,0], trajs[k]])
        plt.plot(pred_plot[:, 0], pred_plot[:, 1], marker='o', linestyle='-',
                 color=colors[k], label=f'Pred {k} (prob: {confs[k]:.2f})',
                 alpha=0.7, linewidth=max(1, int(confs[k]*6)), markersize=5)

    for i in range(len(social_mask)):
        if social_mask[i]:
            plt.plot(social[i, 0], social[i, 1], 'c^', markersize=8)
    if social_mask.any():
        plt.plot([], [], 'c^', label='Neighbors')

    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    # FIX #9: Updated title
    plt.title("SATT v4 Multi-Modal Trajectory Prediction", fontsize=14)
    plt.xlabel("X Displacement (meters)")
    plt.ylabel("Y Displacement (meters)")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'vis_{idx}.png', dpi=150)
    plt.close()
    print(f"Saved visualization to vis_{idx}.png")


# FIX #10: Test-Time Augmentation
def predict_with_tta(model, hist, social, social_mask, n_rotations=8):
    """
    Run the model on N rotated copies of the input and aggregate top-K predictions.
    hist: [1, hist_len, 7], social: [1, max_n, 7], social_mask: [1, max_n]
    Returns: trajs [1, K, fut_len, 2], confs [1, K]
    """
    model.eval()
    all_trajs = []
    all_confs = []

    angles = np.linspace(0, 2*np.pi, n_rotations, endpoint=False)

    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        inv_rot = torch.tensor([[c, s], [-s, c]], dtype=torch.float32)

        # Rotate input
        h_rot = hist.clone()
        h_rot[:, :, :2] = h_rot[:, :, :2] @ rot.T
        h_rot[:, :, 2:4] = h_rot[:, :, 2:4] @ rot.T

        s_rot = social.clone()
        s_rot[:, :, :2] = s_rot[:, :, :2] @ rot.T
        s_rot[:, :, 2:4] = s_rot[:, :, 2:4] @ rot.T

        with torch.no_grad():
            trajs, confs = model(h_rot, s_rot, social_mask)

        # Rotate predictions back to original frame
        B, K, T, _ = trajs.shape
        trajs_flat = trajs.view(-1, 2)
        trajs_back = (trajs_flat @ inv_rot.T).view(B, K, T, 2)

        all_trajs.append(trajs_back)
        all_confs.append(torch.softmax(confs, dim=-1))

    # Stack all: [n_rotations, B, K, T, 2] and [n_rotations, B, K]
    all_trajs = torch.cat(all_trajs, dim=1)   # [B, n_rot*K, T, 2]
    all_confs = torch.cat(all_confs, dim=1)   # [B, n_rot*K]

    # Select top-K by confidence
    K = 3
    topk_confs, topk_idx = all_confs.topk(K, dim=1)
    B = all_trajs.size(0)
    batch_idx = torch.arange(B).unsqueeze(1).expand_as(topk_idx)
    topk_trajs = all_trajs[batch_idx, topk_idx]

    return topk_trajs, topk_confs


def full_evaluation(model, dataset, use_tta=False):
    """Run full evaluation with optional Test-Time Augmentation."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()

    ped_ade, ped_fde = [], []
    bike_ade, bike_fde = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            hist = batch['hist'].float()
            gt_fut = batch['fut'].float()
            social = batch['social'].float()
            social_mask = batch['social_mask'].bool()

            if use_tta:
                trajs, confs = predict_with_tta(model, hist, social, social_mask)
            else:
                trajs, confs = model(hist, social, social_mask)

            m_ade, m_fde, _ = compute_min_ade_fde(trajs, gt_fut, K=3)

            is_cyclist = hist[:, 4, 5].bool()
            for i in range(len(is_cyclist)):
                if is_cyclist[i]:
                    bike_ade.append(m_ade[i].item())
                    bike_fde.append(m_fde[i].item())
                else:
                    ped_ade.append(m_ade[i].item())
                    ped_fde.append(m_fde[i].item())

    avg_ped_ade = np.mean(ped_ade) if ped_ade else 0.0
    avg_ped_fde = np.mean(ped_fde) if ped_fde else 0.0
    avg_bike_ade = np.mean(bike_ade) if bike_ade else 0.0
    avg_bike_fde = np.mean(bike_fde) if bike_fde else 0.0
    avg_total_ade = np.mean(ped_ade + bike_ade)
    avg_total_fde = np.mean(ped_fde + bike_fde)

    tta_label = " (with TTA)" if use_tta else ""
    print("\n" + "="*60)
    print(f"       SATT v4 — Dual-Class Final Results{tta_label}")
    print("="*60)
    print(f"  TOTAL minADE_3:      {avg_total_ade:.4f} meters")
    print(f"  TOTAL minFDE_3:      {avg_total_fde:.4f} meters")
    print("-" * 60)
    print(f"  Pedestrian minADE_3: {avg_ped_ade:.4f} meters  (n={len(ped_ade)})")
    print(f"  Pedestrian minFDE_3: {avg_ped_fde:.4f} meters")
    print("-" * 60)
    print(f"  Bicycle minADE_3:    {avg_bike_ade:.4f} meters  (n={len(bike_ade)})")
    print(f"  Bicycle minFDE_3:    {avg_bike_fde:.4f} meters")
    print("="*60)

    return avg_total_ade, avg_total_fde

if __name__ == '__main__':
    dataset = NuScenesTrajectoryDataset('v1.0-mini/v1.0-mini', split='val', augment=False)
    model = TrajectoryPredictor()
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        print("Loaded best_model.pth successfully")
    except:
        print("Warning: best_model.pth not found, using untrained weights for demo.")

    # Standard evaluation
    print("\n📊 Standard Evaluation:")
    full_evaluation(model, dataset, use_tta=False)

    # TTA evaluation (FIX #10)
    print("\n📊 Test-Time Augmentation Evaluation:")
    full_evaluation(model, dataset, use_tta=True)

    # Generate visualizations
    indices = [5, 15, 30, 50, 80]
    for idx in indices:
        if idx < len(dataset):
            visualize_prediction(model, dataset, idx=idx)
