import torch
from data.dataset import NuScenesTrajectoryDataset
from model.trajectory_predictor import TrajectoryPredictor
import argparse

def main():
    parser = argparse.ArgumentParser("SATT Trajectory Predicton Demo")
    parser.add_argument('--idx', type=int, default=15, help='Dataset sample index to predict')
    args = parser.parse_args()

    # Create dummy config and load best model
    model = TrajectoryPredictor()
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        print("Model loaded.")
    except Exception as e:
        print(f"Running untrained model for demonstration purposes: {e}")
        
    model.eval()
    
    # Load dataset
    print(f"Loading data sample at index {args.idx}...")
    dataset = NuScenesTrajectoryDataset('v1.0-mini/v1.0-mini', split='val', augment=False)
    
    # Extract
    batch = dataset[args.idx]
    
    hist = batch['hist'].float().unsqueeze(0)
    social = batch['social'].float().unsqueeze(0)
    social_mask = batch['social_mask'].bool().unsqueeze(0)
    
    with torch.no_grad():
        trajs, confs = model(hist, social, social_mask)
        
    confs = torch.softmax(confs[0], dim=0).numpy()
    trajs = trajs[0].numpy()
    
    print("\n[Input]")
    print(f"Past 2s Trajectory (agent relative frame):\n {batch['hist'][:, :2].numpy()}")
    print(f"Social Context (neighbors): {social_mask.sum().item()} active")
    
    print("\n[Predictions (next 3s)]")
    for k in range(3):
        print(f"Path {k} (confidence: {confs[k]:.2%} ):\n {trajs[k]}\n")
        
if __name__ == '__main__':
    main()
