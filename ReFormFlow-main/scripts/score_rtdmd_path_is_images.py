"""Score corrected Scheme A images using the identical DMD four-metric protocol."""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from flow_grpo.rewards import aesthetic_score, clip_score, hpsv2_score, pickscore_score


SCORERS = {
    "clipscore": clip_score,
    "aesthetic": aesthetic_score,
    "pickscore": pickscore_score,
    "hpsv2": hpsv2_score,
}


def collect_records(folder, expected_count):
    records = []
    for source in sorted(folder.glob("rank-*/prompts.json")):
        for record in json.loads(source.read_text()):
            image = source.parent / f"{record['index']:06d}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            records.append(dict(record, image=str(image.relative_to(folder))))
    records.sort(key=lambda item: item["index"])
    if [item["index"] for item in records] != list(range(expected_count)):
        raise ValueError(f"Expected unique test indices 0..{expected_count - 1}; found {len(records)} records")
    return records


def write_report(run_dir, folder, step, records, metrics):
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("reward_fn") or not config.get("path_is", {}).get("enabled"):
        raise ValueError("This report requires posterior MH without reward training")
    sample = config["sample"]
    nfe = len(sample["denoising_step_list"][:sample["num_steps"]])
    row = {
        "Method": config["method_name"],
        "NFE": nfe,
        "Post-Train": f"Path-IS lambda={config['posterior_consistency']['weight']:.2f}",
        "CLIP Score": metrics["clipscore"] * 100,
        "Aesthetic Score": metrics["aesthetic"],
        "Pick Score": metrics["pickscore"] * 26,
        "HP Score": metrics["hpsv2"] * 100,
    }
    report = {
        "step": step,
        "num_images": len(records),
        "table": row,
        "raw_repository_means": metrics,
        "protocol": {
            "dataset": str(Path(config["eval"]["dataset"]) / "test.txt"),
            "resolution": config["resolution"],
            "seed": config["eval"]["seed"],
            "ema": config["eval"]["use_ema"],
            "images": "The identical saved PNG files are scored by all four models",
            "clip": "openai/clip-vit-large-patch14; repository logits/100; table = raw * 100",
            "aesthetic": "improved-aesthetic-predictor sac+logos+ava1-l14-linearMSE.pth; no rescaling",
            "pickscore": "yuvalkirstain/PickScore_v1; repository raw/26; table restores raw logits",
            "hp": "HPSv2.1 (xswu/HPSv2 HPS_v2.1_compressed.pt); table = cosine * 100",
            "training_rewards": {},
            "path_is": config["path_is"],
            "posterior_consistency": config["posterior_consistency"],
            "interpretation": "Literal Teacher/Student CPS path ratios, Student shared readout, paired-reference conditional IS; regularization reported separately",
            "nfe": "Generator transformer forward passes per image; excludes training and scoring",
            "comparability": "Scores require the same prompts, model versions, preprocessing and scales for comparison",
        },
    }
    dataset = Path(config["eval"]["dataset"]) / "test.txt"
    if dataset.is_file():
        report["protocol"]["test_file_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    (folder / "final-metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (folder / "final-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    headings = " | ".join(row)
    values = " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row.values())
    text = f"| {headings} |\n| " + " | ".join(["---"] * len(row)) + f" |\n| {values} |\n\n"
    text += f"Evaluation prompt set: {len(records)} images; step {step}; EMA={config['eval']['use_ema']}.\n\n"
    text += "CLIP and HP scores use the ×100 display scale; Pick Score is the unnormalized logit. "
    text += "HP uses HPSv2.1. All four metrics score the same saved images. "
    text += "These values do not establish comparability with a paper using a different evaluation protocol.\n"
    (folder / "final-metrics.md").write_text(text)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.expected_count < 1:
        parser.error("batch-size and expected-count must be positive")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    try:
        folder = args.run_dir / "eval" / f"step-{args.step}"
        records = collect_records(folder, args.expected_count)
        local = [dict(record, scores={}) for record in records[rank::world_size]]
        for name, factory in SCORERS.items():
            print(f"rank={rank} loading evaluation model: {name}", flush=True)
            scorer = factory(device)
            with torch.no_grad():
                for start in range(0, len(local), args.batch_size):
                    batch = local[start:start + args.batch_size]
                    pixels = []
                    for record in batch:
                        with Image.open(folder / record["image"]) as image:
                            pixels.append(np.array(image.convert("RGB"), copy=True))
                    images = torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).to(device).float() / 255
                    values, _ = scorer(images, [r["prompt"] for r in batch], [r.get("metadata", {}) for r in batch])
                    if torch.is_tensor(values):
                        values = values.detach().float().cpu().numpy()
                    values = np.asarray(values).reshape(-1)
                    if len(values) != len(batch) or not np.isfinite(values).all():
                        raise ValueError(f"Invalid {name} scores at batch {start}")
                    for record, value in zip(batch, values):
                        record["scores"][name] = float(value)
            del scorer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"rank={rank} {name} completed: {len(local)} images", flush=True)
        part = folder / f"scores-rank-{rank}.json"
        part.write_text(json.dumps(local, indent=2, ensure_ascii=False))
        gathered = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(gathered, local)
        else:
            gathered[0] = local
        if rank == 0:
            combined = sorted([record for chunk in gathered for record in chunk], key=lambda r: r["index"])
            if [r["index"] for r in combined] != list(range(args.expected_count)):
                raise ValueError("Distributed scoring did not cover the test set exactly once")
            metrics = {name: float(np.mean([r["scores"][name] for r in combined], dtype=np.float64)) for name in SCORERS}
            (folder / "per-image-scores.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))
            write_report(args.run_dir, folder, args.step, combined, metrics)
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
