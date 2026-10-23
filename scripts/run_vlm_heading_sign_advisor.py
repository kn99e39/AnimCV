#!/usr/bin/env python3
"""EXPERIMENT (Worklog 92): run the frozen Qwen3-VL-8B torso up/down advisor on the invoked heading rows.

Population (runtime only, no GT): every 3DPW VALIDATION / TEST and MPI-INF-3DHP TEST row where the GT-free W91 heading
exists and ``motion.vlm_heading_sign_advisor.should_invoke`` holds (S0 torso inclination >= 45 deg, valid detector
torso).  Jobs per invoked row: the real advisor image with both prompts (option orders), and a shuffled-image control
(the image of an invoked row from another sequence, ``back_first`` prompt).  Responses go to a resumable JSONL keyed
by (kind, prompt, sample_id) with image / crop / prompt / model fingerprints.  GT never enters any input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank
from framepose.crops import crop_box, render_crop
from framepose.mpi3dhp_test_transfer import FPS
from framepose.vlm_depth_advisor import shuffled_donors
from motion.runtime_heading import runtime_heading
from motion.vlm_heading_sign_advisor import (
    ADVISOR_RESOLUTION, GENERATION, MODEL_REPOSITORY, MODEL_REVISION, PROMPT_SHA256, PROMPTS, should_invoke,
    torso_axis_x, torso_inclination_degrees,
)
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split

S0_SHA = {"validation": "402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9",
          "test": "4357c7bdb46d81e7712269b08c45f39452d7a46e697a0c26fdb70d65fe1f4996",
          "3dhp": "a78adeb0ed0391b26d4c378e0ea15ce7d1e1eaf71dfcd9a384bb6dd6e3c6e420"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hp_shim(mx):
    samples = [SimpleNamespace(sequence_id=s.sequence_id, frame_index=s.frame_index, fps=float(FPS[s.sequence_id.split(":")[1]]),
                               timestamp=s.frame_index / float(FPS[s.sequence_id.split(":")[1]]), image_size=s.image_size,
                               image_reference=s.image_reference, sample_id=s.sample_id) for s in mx.samples]
    return SimpleNamespace(samples=samples, arrays=mx.arrays)


def population(name, view, preds, seqs, root):
    out = []
    for sid in seqs:
        pos = sequence_frame_positions(view, sid)
        lifted = build_h0_lifted_sequence(view, preds, sid)
        heads = runtime_heading(lifted)
        for i, p in enumerate(pos):
            s = view.samples[p]
            inc = torso_inclination_degrees(lifted.frames[i])
            ax = torso_axis_x(view.arrays["input_2d"][p], view.arrays["input_valid"][p], s.image_size)
            if heads[i].yaw_radians is None or not should_invoke(inc, ax):
                continue
            out.append({"population": name, "position": int(p), "sample_id": s.sample_id, "sequence_id": sid,
                        "frame_index": int(s.frame_index), "yaw_s0": heads[i].yaw_radians, "w91_status": heads[i].status,
                        "inclination_s0": inc, "axis_x": ax, "root": root, "image": s.image_reference.relative_path,
                        "image_size": list(s.image_size)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "s0_dir", "hp_bank", "hp_s0", "img3dpw", "img3dhp", "snapshot", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    import cv2
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if args.snapshot.name != MODEL_REVISION:
        raise SystemExit("STOP: snapshot directory is not the pinned revision")
    for split in ("validation", "test"):
        if _sha((args.s0_dir / f"prediction_{split}.npy").read_bytes()) != S0_SHA[split]:
            raise SystemExit("STOP: S0 3DPW predictions changed")
    if _sha(args.hp_s0.read_bytes()) != S0_SHA["3dhp"]:
        raise SystemExit("STOP: S0 3DHP predictions changed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    mx = ExternalBank.load(args.hp_bank)
    hv = hp_shim(mx)
    rows = []
    for split in ("validation", "test"):
        rows += population(f"3dpw_{split}", bank, s0, sequence_ids_in_split(bank, split), "3dpw")
    rows += population("3dhp", hv, np.load(args.hp_s0), sorted({s.sequence_id for s in mx.samples}), "3dhp")
    donors = shuffled_donors([(r["sequence_id"], r["frame_index"]) for r in rows])
    (args.out_dir / "population.json").write_text(json.dumps({"rows": rows, "donor_sample_id": {r["sample_id"]: rows[d]["sample_id"] for r, d in zip(rows, donors)}}))
    jobs = [("real", k, r, r) for r in rows for k in PROMPTS] + [("shuffled", "back_first", r, rows[d]) for r, d in zip(rows, donors)]
    path = args.out_dir / "responses.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            it = json.loads(line)
            done.add((it["kind"], it["prompt"], it["sample_id"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2]["sample_id"]) not in done]
    processor = AutoProcessor.from_pretrained(args.snapshot)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.snapshot, device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=False))
    model.eval()
    (args.out_dir / "model.json").write_text(json.dumps({"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
                                                         "transformers": transformers.__version__, "torch": torch.__version__,
                                                         "device": torch.cuda.get_device_name(0), "generation": GENERATION,
                                                         "prompt_sha256": PROMPT_SHA256, "resolution": ADVISOR_RESOLUTION}, indent=1))

    def advisor_image(r):
        root = args.img3dpw if r["root"] == "3dpw" else args.img3dhp
        data = (root / r["image"]).read_bytes()
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        arrays = bank.arrays if r["root"] == "3dpw" else mx.arrays
        box = crop_box(arrays["input_2d"][r["position"]], arrays["input_valid"][r["position"]], tuple(r["image_size"]))
        crop = render_crop(rgb, box, ADVISOR_RESOLUTION)
        return Image.fromarray(crop), _sha(data), _sha(crop.tobytes())

    started = time.time()
    with path.open("a") as out:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start + args.batch_size]
            images, recs = [], []
            for kind, prompt, target, source in batch:
                img, image_sha, crop_sha = advisor_image(source)
                images.append(img)
                recs.append({"kind": kind, "prompt": prompt, "sample_id": target["sample_id"], "image_source_sample_id": source["sample_id"],
                             "image_sha256": image_sha, "advisor_image_sha256": crop_sha, "prompt_sha256": PROMPT_SHA256[prompt]})
            msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[j[1]]}]}] for j in batch]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**inputs, **GENERATION)
            dt = (time.time() - t0) / len(batch)
            for rec, text in zip(recs, processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)):
                rec.update({"raw_text": text, "seconds_per_image": dt})
                out.write(json.dumps(rec) + "\n")
            out.flush()
            if (start // args.batch_size) % 50 == 0:
                print(json.dumps({"done": start + len(batch), "of": len(jobs), "seconds": round(time.time() - started, 1)}), flush=True)
    print(json.dumps({"finished": len(jobs), "seconds": round(time.time() - started, 1), "invoked_rows": len(rows)}))


if __name__ == "__main__":
    main()
