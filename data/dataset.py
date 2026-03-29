import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import math
from collections import defaultdict

class NuScenesTrajectoryDataset(Dataset):
    def __init__(self, data_path, split='train', seq_len=11, hist_len=5, fut_len=6, augment=False):
        self.data_path = data_path
        self.split = split
        self.seq_len = seq_len
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.augment = augment

        with open(os.path.join(data_path, 'sample_annotation.json'), 'r') as f:
            self.annotations = json.load(f)
        with open(os.path.join(data_path, 'instance.json'), 'r') as f:
            self.instances = json.load(f)
        with open(os.path.join(data_path, 'category.json'), 'r') as f:
            self.categories = json.load(f)
        with open(os.path.join(data_path, 'sample.json'), 'r') as f:
            self.samples = json.load(f)
        with open(os.path.join(data_path, 'attribute.json'), 'r') as f:
            self.attributes = json.load(f)
        with open(os.path.join(data_path, 'scene.json'), 'r') as f:
            self.scenes = json.load(f)

        self.cat_lookup = {c['token']: c['name'] for c in self.categories}
        self.attr_lookup = {a['token']: a['name'] for a in self.attributes}
        self.sample_lookup = {s['token']: s for s in self.samples}
        self.valid_cats = {c['token'] for c in self.categories if 'pedestrian' in c['name'] or 'bicycle' in c['name']}
        self.ann_lookup = {a['token']: a for a in self.annotations}
        self.instances_dict = {i['token']: i for i in self.instances}

        self.scene_samples = defaultdict(list)
        for s in self.samples:
            self.scene_samples[s['scene_token']].append(s['token'])

        scene_tokens = sorted(self.scene_samples.keys())
        if split == 'train':
            self.valid_scenes = set(scene_tokens[:8])
        else:
            self.valid_scenes = set(scene_tokens[8:])

        self.sample_to_anns = defaultdict(list)
        for a in self.annotations:
            self.sample_to_anns[a['sample_token']].append(a)

        self._extract_trajectories()

    def _extract_trajectories(self):
        self.sequences = []
        instances_to_process = [i for i in self.instances if i['category_token'] in self.valid_cats]

        for inst in instances_to_process:
            curr_token = inst['first_annotation_token']
            full_seq = []
            while curr_token:
                ann = self.ann_lookup[curr_token]
                samp = self.sample_lookup[ann['sample_token']]
                if samp['scene_token'] not in self.valid_scenes:
                    curr_token = ann.get('next', '')
                    continue
                full_seq.append(ann)
                curr_token = ann.get('next', '')

            if len(full_seq) >= self.seq_len:
                for i in range(len(full_seq) - self.seq_len + 1):
                    window = full_seq[i:i+self.seq_len]
                    self.sequences.append(window)

    def _get_agent_features(self, ann, prev_ann=None):
        x, y, _ = ann['translation']
        if prev_ann:
            px, py, _ = prev_ann['translation']
            dx, dy = x - px, y - py
        else:
            dx, dy = 0.0, 0.0

        q = ann['rotation']
        heading = np.arctan2(2.0 * (q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))

        cat_name = self.cat_lookup[self.instances_dict[ann['instance_token']]['category_token']]
        is_cyclist = 1.0 if 'bicycle' in cat_name else 0.0

        is_moving = 0.0
        if 'attribute_tokens' in ann and len(ann['attribute_tokens']) > 0:
            attr_name = self.attr_lookup[ann['attribute_tokens'][0]]
            if 'moving' in attr_name:
                is_moving = 1.0

        return np.array([x, y, dx, dy, heading, is_cyclist, is_moving], dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def _normalize_angle(a):
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def __getitem__(self, idx):
        window = self.sequences[idx]

        features = []
        for i, ann in enumerate(window):
            prev_ann = window[i-1] if i > 0 else None
            feat = self._get_agent_features(ann, prev_ann)
            features.append(feat)

        features = np.stack(features)  # [seq_len, 7]

        # Origin shift
        origin_idx = self.hist_len - 1
        origin_x = features[origin_idx, 0].copy()
        origin_y = features[origin_idx, 1].copy()
        features[:, 0] -= origin_x
        features[:, 1] -= origin_y

        # FIX #2: Strict Rotation Invariance (+Y axis alignment)
        origin_heading = features[origin_idx, 4]
        align_angle = np.pi/2.0 - origin_heading
        c_a, s_a = np.cos(align_angle), np.sin(align_angle)
        rot_mat_align = np.array([[c_a, -s_a], [s_a, c_a]], dtype=np.float32)

        features[:, :2] = features[:, :2] @ rot_mat_align.T
        features[:, 2:4] = features[:, 2:4] @ rot_mat_align.T
        features[:, 4] += align_angle
        # FIX #2b: Normalize heading to [-π, π]
        features[:, 4] = self._normalize_angle(features[:, 4])

        # Augmentation (ONLY on history, NOT on future GT)
        if self.augment and self.split == 'train':
            # FIX #3: Small jitter rotation (±15°) to preserve partial heading invariance
            angle = np.random.uniform(-np.pi/12, np.pi/12)
            c, s = np.cos(angle), np.sin(angle)
            rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)

            features[:, :2] = features[:, :2] @ rot_mat.T
            features[:, 2:4] = features[:, 2:4] @ rot_mat.T
            features[:, 4] = self._normalize_angle(features[:, 4] + angle)

            # Flip horizontally
            if np.random.rand() > 0.5:
                features[:, 0] *= -1
                features[:, 2] *= -1
                features[:, 4] = self._normalize_angle(np.pi - features[:, 4])

            # FIX #1: Gaussian noise ONLY on history coordinates, NOT future GT
            noise = np.random.normal(0, 0.05, size=(self.hist_len, 2)).astype(np.float32)
            features[:self.hist_len, :2] += noise

        # Split
        hist = features[:self.hist_len]        # [hist_len, 7]
        fut = features[self.hist_len:, :2]     # [fut_len, 2]

        # Social Context
        target_sample = window[origin_idx]['sample_token']
        target_inst = window[origin_idx]['instance_token']
        neighbors_anns = [a for a in self.sample_to_anns[target_sample] if a['instance_token'] != target_inst]

        social_features = []
        for n_ann in neighbors_anns:
            n_x, n_y, _ = n_ann['translation']
            rel_x, rel_y = n_x - origin_x, n_y - origin_y
            dist = np.sqrt(rel_x**2 + rel_y**2)
            if dist < 15.0:
                n_feat = self._get_agent_features(n_ann, None)
                n_feat[0] = rel_x
                n_feat[1] = rel_y
                social_features.append((dist, n_feat))

        if len(social_features) == 0:
            social_features_arr = np.zeros((1, 7), dtype=np.float32)
        else:
            social_features.sort(key=lambda x: x[0])
            social_features_arr = np.stack([feat for _, feat in social_features])
            # Apply +Y rotation alignment to neighbors
            social_features_arr[:, :2] = social_features_arr[:, :2] @ rot_mat_align.T
            social_features_arr[:, 2:4] = social_features_arr[:, 2:4] @ rot_mat_align.T
            social_features_arr[:, 4] = self._normalize_angle(social_features_arr[:, 4] + align_angle)

        MAX_NEIGHBORS = 20
        num_neighbors = min(len(social_features_arr), MAX_NEIGHBORS)
        padded_social = np.zeros((MAX_NEIGHBORS, 7), dtype=np.float32)
        padded_social[:num_neighbors] = social_features_arr[:num_neighbors]
        social_mask = np.zeros((MAX_NEIGHBORS,), dtype=np.bool_)
        social_mask[:num_neighbors] = True

        return {
            'hist': torch.tensor(hist),
            'fut': torch.tensor(fut),
            'social': torch.tensor(padded_social),
            'social_mask': torch.tensor(social_mask)
        }

if __name__ == '__main__':
    ds = NuScenesTrajectoryDataset('v1.0-mini/v1.0-mini', augment=True)
    print(f"Train dataset size: {len(ds)} sequences")
    if len(ds) > 0:
        batch = ds[0]
        print(f"Hist shape: {batch['hist'].shape}")
        print(f"Fut shape: {batch['fut'].shape}")
        print(f"Social shape: {batch['social'].shape}")
