"""DIAGNOSTIC (Worklog 73): OpenPose COCO-18 body detector for external RGB.

The FrameBank benchmark observation for 3DPW is the dataset-shipped OpenPose
COCO-18 detection (``official_3dpw_2d_detection``).  AnimCV never ran that
detector, so for an external source the closest reproducible equivalent is
the CMU OpenPose COCO body model run here: network definition from
Hzzone/pytorch-openpose (pinned commit, mounted at ``OPENPOSE_ROOT``) and the
converted CMU COCO weights (sha256 pinned).  Inference parameters are the
OpenPose defaults, fixed before any use: single scale, network input height
368, stride 8, part threshold 0.1, PAF threshold 0.05, 10 PAF samples, peak
smoothing sigma 3, and person pruning at < 4 parts or mean score < 0.4.

Peak finding and PAF sampling run on the GPU (bicubic upsampling, separable
Gaussian, 4-neighbour non-maximum test); person assembly is the reference
OpenPose greedy algorithm on the CPU.  Keypoint confidences are the raw
heat-map scores at the peaks, as in OpenPose.  No GT is read here.

``to_canonical_input`` converts one detected person to the FrameBank input
layout with the exact 3DPW adapter rules (``three_dpw_adapter._canonical_2d``
plus the ``build_dataset`` thorax->neck alias and image-size normalization).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from pose.pose_lifter import H36M_NAMES

PARAMS = {"net_height": 368, "stride": 8, "pad_value": 128, "part_threshold": 0.1, "paf_threshold": 0.05,
          "paf_samples": 10, "gaussian_sigma": 3.0, "min_parts": 4, "min_mean_score": 0.4}
WEIGHTS_SHA256 = "25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746"
OPENPOSE_COMMIT = "5ee71dc10020403dc3def2bb68f9b77c40337ae2"
COCO18 = ("nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
          "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
          "right_eye", "left_eye", "right_ear", "left_ear")
# OpenPose COCO limb sequence (1-based part ids) and PAF channel pairs (19-based).
LIMB_SEQ = ((2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13),
            (13, 14), (2, 1), (1, 15), (15, 17), (1, 16), (16, 18), (3, 17), (6, 18))
MAP_IDX = ((31, 32), (39, 40), (33, 34), (35, 36), (41, 42), (43, 44), (19, 20), (21, 22), (23, 24), (25, 26),
           (27, 28), (29, 30), (47, 48), (49, 50), (53, 54), (51, 52), (55, 56), (37, 38), (45, 46))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OpenPoseCOCO18:
    def __init__(self, device: str = "cuda", root: str | None = None):
        import torch

        root = root or os.environ["OPENPOSE_ROOT"]
        weights = Path(root) / "body_pose_model.pth"
        if file_sha256(weights) != WEIGHTS_SHA256:
            raise SystemExit("OpenPose COCO-18 weights sha256 mismatch")
        sys.path.insert(0, root)
        from src.model import bodypose_model  # noqa: E402  (pinned Hzzone network definition)

        self.torch = torch
        self.device = torch.device(device)
        model = bodypose_model()
        raw = torch.load(weights, map_location="cpu")
        # Hzzone's ``util.transfer``: model keys are "<block>.<layer>.<param>", weights "<layer>.<param>".
        model.load_state_dict({k: raw[".".join(k.split(".")[1:])] for k in model.state_dict()})
        self.model = model.to(self.device).eval()
        sigma = PARAMS["gaussian_sigma"]
        radius = int(4.0 * sigma + 0.5)
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        self.kernel = (kernel / kernel.sum()).to(self.device)
        self.radius = radius

    def identity(self) -> dict:
        return {"detector": "OpenPose COCO-18 body model (CMU weights, PyTorch port)",
                "network_definition": f"Hzzone/pytorch-openpose@{OPENPOSE_COMMIT} src/model.py bodypose_model",
                "weights_sha256": WEIGHTS_SHA256, "params": dict(PARAMS), "torch": self.torch.__version__}

    def maps(self, bgr: np.ndarray):
        """Heat maps (19,H,W) and PAFs (38,H,W) at original image resolution, on device."""
        import cv2

        torch, F = self.torch, self.torch.nn.functional
        h, w = bgr.shape[:2]
        scale = PARAMS["net_height"] / h
        img = cv2.resize(bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        stride = PARAMS["stride"]
        ph, pw = (-img.shape[0]) % stride, (-img.shape[1]) % stride
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=PARAMS["pad_value"])
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 256 - 0.5)
        with torch.no_grad():
            paf, heat = self.model(x.to(self.device))
            out = torch.cat([heat, paf], 1)
            out = F.interpolate(out, scale_factor=stride, mode="bicubic", align_corners=False)
            out = out[:, :, :img.shape[0] - ph, :img.shape[1] - pw]
            out = F.interpolate(out, size=(h, w), mode="bicubic", align_corners=False)[0]
        return out[:19], out[19:]

    def peaks(self, heat):
        torch, F = self.torch, self.torch.nn.functional
        maps = heat[:18][:, None]
        k = self.kernel
        blurred = F.conv2d(F.pad(maps, (self.radius, self.radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        blurred = F.conv2d(F.pad(blurred, (0, 0, self.radius, self.radius), mode="reflect"), k.view(1, 1, -1, 1))[:, 0]
        b = blurred
        left = torch.zeros_like(b); left[:, 1:, :] = b[:, :-1, :]
        right = torch.zeros_like(b); right[:, :-1, :] = b[:, 1:, :]
        up = torch.zeros_like(b); up[:, :, 1:] = b[:, :, :-1]
        down = torch.zeros_like(b); down[:, :, :-1] = b[:, :, 1:]
        binary = (b >= left) & (b >= right) & (b >= up) & (b >= down) & (b > PARAMS["part_threshold"])
        part, ys, xs = torch.nonzero(binary, as_tuple=True)
        scores = heat[part, ys, xs]
        part, ys, xs, scores = (t.cpu().numpy() for t in (part, ys, xs, scores))
        all_peaks, counter = [], 0
        for p in range(18):
            sel = np.nonzero(part == p)[0]
            sel = sel[np.lexsort((xs[sel], ys[sel]))]  # numpy nonzero order (row-major)
            all_peaks.append([(int(xs[i]), int(ys[i]), float(scores[i]), counter + n) for n, i in enumerate(sel)])
            counter += len(sel)
        return all_peaks

    def __call__(self, bgr: np.ndarray) -> list[dict]:
        heat, paf = self.maps(bgr)
        all_peaks = self.peaks(heat)
        h = bgr.shape[0]
        torch = self.torch
        connection_all, special_k = [], []
        mid = PARAMS["paf_samples"]
        for k, (limb, chans) in enumerate(zip(LIMB_SEQ, MAP_IDX)):
            cand_a, cand_b = all_peaks[limb[0] - 1], all_peaks[limb[1] - 1]
            if not cand_a or not cand_b:
                special_k.append(k)
                connection_all.append([])
                continue
            a = np.array([c[:2] for c in cand_a], float)
            b = np.array([c[:2] for c in cand_b], float)
            vec = b[None] - a[:, None]
            norm = np.maximum(np.linalg.norm(vec, axis=2), 0.001)
            unit = vec / norm[..., None]
            t = np.linspace(0, 1, mid)
            pts = a[:, None, None] + vec[:, :, None] * t[None, None, :, None]
            # np.round matches Python's round-half-even used by the reference int(round(.)).
            idx = torch.from_numpy(np.round(pts).astype(np.int64)).to(self.device)
            sx = paf[chans[0] - 19][idx[..., 1], idx[..., 0]].cpu().numpy()
            sy = paf[chans[1] - 19][idx[..., 1], idx[..., 0]].cpu().numpy()
            score_mid = sx * unit[..., 0:1] + sy * unit[..., 1:2]
            with_prior = score_mid.mean(axis=2) + np.minimum(0.5 * h / norm - 1, 0)
            crit1 = (score_mid > PARAMS["paf_threshold"]).sum(axis=2) > 0.8 * mid
            candidates = [[i, j, with_prior[i, j], with_prior[i, j] + cand_a[i][2] + cand_b[j][2]]
                          for i in range(len(cand_a)) for j in range(len(cand_b))
                          if crit1[i, j] and with_prior[i, j] > 0]
            candidates.sort(key=lambda c: c[2], reverse=True)
            connection = np.zeros((0, 5))
            for i, j, s, _ in candidates:
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                    if len(connection) >= min(len(cand_a), len(cand_b)):
                        break
            connection_all.append(connection)

        candidate = np.array([p for peaks in all_peaks for p in peaks], float).reshape(-1, 4)
        subset = -1 * np.ones((0, 20))
        for k in range(len(MAP_IDX)):
            if k in special_k:
                continue
            part_as, part_bs = connection_all[k][:, 0], connection_all[k][:, 1]
            index_a, index_b = np.array(LIMB_SEQ[k]) - 1
            for i in range(len(connection_all[k])):
                found, subset_idx = 0, [-1, -1]
                for j in range(len(subset)):
                    if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                        subset_idx[found] = j
                        found += 1
                if found == 1:
                    j = subset_idx[0]
                    if subset[j][index_b] != part_bs[i]:
                        subset[j][index_b] = part_bs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:
                        subset[j1][:-2] += subset[j2][:-2] + 1
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][index_b] = part_bs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[index_a], row[index_b] = part_as[i], part_bs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
        keep = [i for i in range(len(subset))
                if not (subset[i][-1] < PARAMS["min_parts"] or subset[i][-2] / subset[i][-1] < PARAMS["min_mean_score"])]
        people = []
        for row in subset[keep]:
            kp = np.zeros((18, 3))
            for part in range(18):
                if row[part] >= 0:
                    kp[part] = candidate[int(row[part]), :3]
            people.append({"keypoints": kp, "score": float(row[-2]), "parts": int(row[-1])})
        return people


def to_canonical_input(keypoints: np.ndarray, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(18,3) pixel COCO-18 -> FrameBank (input_2d (17,3) [x/W, y/H, conf], visible (17,)).

    Exactly the 3DPW path: ``three_dpw_adapter._canonical_2d`` then the
    ``temporal_lifter.build_dataset`` row rule (thorax reads the neck landmark;
    coordinates are written even for an invisible midpoint, whose confidence is
    0).  ``visible`` is the landmark flag that ``build_dataset`` ANDs into
    ``target_valid``.
    """
    from pose.three_dpw_adapter import _canonical_2d

    landmarks = _canonical_2d(np.asarray(keypoints, float).T)
    width, height = image_size
    out = np.zeros((len(H36M_NAMES), 3), np.float32)
    visible = np.zeros(len(H36M_NAMES), bool)
    for j, name in enumerate(H36M_NAMES):
        lm = landmarks.get("neck" if name == "thorax" else name)
        if lm is not None:
            out[j] = (lm.x / width, lm.y / height, lm.confidence)
            visible[j] = lm.visible
    return out, visible
