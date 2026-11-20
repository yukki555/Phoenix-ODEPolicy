"""Episode-safe kitchen windows and the ARX X5 canonical80 mapping."""
from bisect import bisect_right

import numpy as np
import torch
import torch.nn.functional as F

ACTIVE = [0, 1, 2, 3, 4, 5, 16, 40, 41, 42, 43, 44, 45, 56]
ARMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPERS = [6, 13]


class KitchenWindows:
    def __init__(self, data_root, sources, split, horizon):
        import zarr

        if split not in {"train", "val"}:
            raise ValueError("Unknown split")
        self.horizon, self.episodes, self.ends = horizon, [], []
        for src in sources:
            group = zarr.open_group(str(data_root / src["name"] / "replay_buffer.zarr"), mode="r")
            ends = np.asarray(group["meta/episode_ends"][:], dtype=int)
            if len(ends) != src["episodes"] or np.any(np.diff(np.r_[0, ends]) <= 0):
                raise ValueError(f"Episode layout changed in {src['name']}")
            for ep, (begin, end) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
                if (ep in src["validation"]) != (split == "val"):
                    continue
                count = int(end-begin-horizon)
                if count > 0:
                    self.episodes.append((group, src["name"], src["task"], ep, int(begin), int(end)))
                    self.ends.append(count + (self.ends[-1] if self.ends else 0))

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        i = bisect_right(self.ends, index)
        g, name, task, ep, begin, _ = self.episodes[i]
        k = begin + index - (self.ends[i-1] if i else 0)
        states = np.asarray(g["data/state_30hz"][k:k+self.horizon+1, :14], dtype=np.float32)
        targets = np.asarray(g["data/action_30hz"][k:k+self.horizon, :14], dtype=np.float32)
        reference = states[0] if k == begin else np.asarray(g["data/action_30hz"][k-1, :14])
        if not np.array_equal(targets, states[1:]) or not np.isfinite(states).all():
            raise ValueError(f"Invalid action/state alignment: {name}/{ep}/{k}")
        frame = np.asarray(g["data/camera"][k])
        if frame.ndim != 3 or frame.dtype != np.uint8 or frame.shape[-1] != 9:
            raise ValueError("Expected main/left/right RGB packed along the last dimension")
        rgb = torch.from_numpy(frame.copy()).reshape(*frame.shape[:2], 3, 3).permute(2, 3, 0, 1)
        return task, rgb, torch.from_numpy(states[:-1]), torch.from_numpy(reference.copy()).float(), torch.from_numpy(targets)

    def fit_train_bounds(self):
        rows = []
        for g, _, _, _, begin, end in self.episodes:
            rows.append(np.asarray(g["data/action_30hz"][begin:end-1, :14], dtype=np.float32))
        bounds = np.quantile(np.concatenate(rows), [0.01, 0.99], axis=0)
        if not np.isfinite(bounds).all():
            raise ValueError("Nonfinite training normalizer")
        return torch.tensor(bounds, dtype=torch.float32)


class JointCoordinates:
    def __init__(self, bounds, device):
        lo, hi = bounds.to(device)
        span = hi - lo
        self.scale = torch.where(span >= 1e-4, 2 / span, 1)
        self.offset = torch.where(span >= 1e-4, -1 - self.scale * lo, -lo)

    def normalized(self, physical):
        return physical.to(self.scale.device) * self.scale + self.offset

    def physical(self, normalized):
        return (normalized - self.offset) / self.scale


def observation(rgb, current, text):
    device = current.device
    image = F.interpolate(rgb.float(), size=(256, 256), mode="bicubic",
                          align_corners=False, antialias=True).clamp_(0, 255)
    state = current.new_zeros(1, 80)
    state[:, ACTIVE] = current.clamp(-5, 5)
    pad = torch.ones((1, 80), dtype=torch.bool, device=device)
    pad[:, ACTIVE] = False
    context, mask = text
    return dict(input_image=(image / 127.5 - 1)[None].to(device), proprio=state,
                action_dim_is_pad=pad, proprio_dim_is_pad=pad,
                context=context[None], context_mask=mask[None],
                camera_is_pad=torch.zeros((1, 3), dtype=torch.bool, device=device),
                embodiment_ids=torch.tensor([2], device=device),
                video_layout="latent_horizontal", video_view_names="main|left|right")
