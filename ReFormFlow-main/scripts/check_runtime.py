"""Inspect local runtime resources without importing scorers or loading weights."""

import argparse
import importlib.metadata as metadata
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def check_model(path):
    root = Path(path).expanduser()
    errors = []
    if not (root / "model_index.json").is_file():
        return [f"Local Diffusers model_index.json not found: {root}"]
    index = json.loads((root / "model_index.json").read_text())
    for name, value in index.items():
        if name.startswith("_") or not isinstance(value, list) or not value or value[0] is None:
            continue
        folder = root / name
        if not folder.is_dir():
            errors.append(f"Missing model component: {folder}")
            continue
        if name in ("transformer", "vae") or name.startswith("text_encoder"):
            if not (folder / "config.json").is_file():
                errors.append(f"Missing model config: {folder}")
            weights = list(folder.glob("*.safetensors")) + list(folder.glob("*.bin"))
            if not any(p.is_file() for p in weights):
                errors.append(f"Missing local model weights: {folder}")
            for manifest in folder.glob("*.index.json"):
                for filename in set(json.loads(manifest.read_text()).get("weight_map", {}).values()):
                    if not (folder / filename).is_file():
                        errors.append(f"Missing weight shard: {folder / filename}")
    return errors


def check_hf_cache(cache, model, required):
    root = Path(cache) / ("models--" + model.replace("/", "--"))
    reference = root / "refs/main"
    if not reference.is_file():
        return [f"Missing cached main reference for {model}: {reference}"]
    snapshot = root / "snapshots" / reference.read_text().strip()
    return [f"Missing cached file for {model}: {name}" for name in required if not (snapshot / name).is_file()]


def package_errors(names):
    errors = []
    for name in names:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}")
    return errors


def check_rewards(rewards, root, hf_cache, home):
    errors = []
    known = {"pickscore", "clipscore", "hpsv2", "geneval", "ocr", "aesthetic", "imagereward"}
    for reward in rewards:
        if reward not in known:
            errors.append(f"Unknown reward: {reward}")
            continue
        if reward == "pickscore":
            errors += check_hf_cache(hf_cache, "yuvalkirstain/PickScore_v1", ["config.json", "model.safetensors"])
            errors += check_hf_cache(hf_cache, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                                     ["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
        if reward in ("clipscore", "aesthetic"):
            errors += check_hf_cache(hf_cache, "openai/clip-vit-large-patch14",
                                     ["config.json", "model.safetensors", "preprocessor_config.json", "vocab.json", "merges.txt"])
        files = []
        if reward == "aesthetic":
            files = ["sac+logos+ava1-l14-linearMSE.pth"]
        elif reward == "hpsv2":
            errors += package_errors(["hpsv2x"])
            files = ["HPS_v2.1_compressed.pt", "open_clip_pytorch_model.bin"]
        elif reward == "geneval":
            errors += package_errors(["mmdet", "open-clip-torch", "clip-benchmark"])
            files = ["mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"]
            try:
                version = metadata.version("mmdet")
                if version.split(".")[0] != "2":
                    errors.append(f"GenEval release requires MMDetection 2.x; installed {version}. No automatic API migration.")
            except metadata.PackageNotFoundError:
                version = None
            spec = PathFinder.find_spec("mmdet", sys.path)
            if spec is not None and spec.origin:
                config = Path(spec.origin).parent.parent / "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
                if not config.is_file():
                    errors.append(f"Missing GenEval detector config: {config}")
        elif reward == "ocr":
            errors += package_errors(["paddleocr", "python-Levenshtein"])
            for component in ("whl/det/en/en_PP-OCRv3_det_infer", "whl/rec/en/en_PP-OCRv4_rec_infer"):
                for filename in ("inference.pdmodel", "inference.pdiparams"):
                    path = Path(home) / ".paddleocr" / component / filename
                    if not path.is_file():
                        errors.append(f"Missing OCR cache: {path}")
        elif reward == "imagereward":
            errors += package_errors(["image-reward"])
            errors.append("ImageReward cache is not validated by this checker; prepare and verify it separately")
        for filename in files:
            path = Path(root) / "reward_ckpts" / filename
            if not path.is_file():
                errors.append(f"Missing reward checkpoint: {path}")
    return errors


def check_imports(root, method):
    root = Path(root).resolve()
    errors = []
    packages = PathFinder.find_spec("flow_grpo", sys.path)
    if packages is None:
        return ["flow_grpo is not on the import path; source jobs/local_env.sh with REFORMDMD_ROOT set"]
    names = ["rewards", "reward_ckpt_path"]
    names += ["stat_tracking_reform"] if method == "rl" else []
    for name in names:
        spec = PathFinder.find_spec(f"flow_grpo.{name}", packages.submodule_search_locations)
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(root):
            errors.append(f"Module flow_grpo.{name} does not resolve inside {root}")
    # Resolve the namespace relative to its search paths without importing its parent.
    patches = PathFinder.find_spec("diffusers_patch", packages.submodule_search_locations)
    expected = ["pipeline_with_logprob", "solver"] if method == "rl" else ["sd3_pipeline_dmd", "solver", "train_dreambooth_lora_sd3"]
    for name in expected:
        spec = None if patches is None else PathFinder.find_spec(name, patches.submodule_search_locations)
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(root):
            errors.append(f"Module flow_grpo.diffusers_patch.{name} does not resolve inside {root}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["rl", "dmd"], default="dmd")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rewards", nargs="*", default=[])
    args = parser.parse_args()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    errors = package_errors(["torch", "diffusers", "peft", "transformers", "ml_collections"])
    errors += check_imports(REPO_ROOT, args.method)
    errors += check_model(args.model)
    errors += check_rewards(args.rewards, REPO_ROOT, cache, Path.home())
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Local files and module origins checked. CUDA execution, scores and memory remain unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
