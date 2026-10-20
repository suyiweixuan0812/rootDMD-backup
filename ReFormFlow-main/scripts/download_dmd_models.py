"""Download verified public SD3 weights and the two PickScore HF snapshots.

Large weights use independent HTTP byte ranges and are checked against the
publisher's SHA256. No authentication credentials are written by this script.
"""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("DMD_MODEL_STORAGE", "/dev/shm/reformdmd-models"))
CACHE = Path(os.environ.get("HF_HOME", "/rootDMD/hf-cache")) / "hub"
WORKERS = int(os.environ.get("DMD_DOWNLOAD_WORKERS", "64"))
CHUNK = 8 * 1024 * 1024
LOCK = threading.Lock()
PROGRESS = dict(bytes=0, chunks=0, total=0)
THREAD_LOCAL = threading.local()


def session():
    if not hasattr(THREAD_LOCAL, "session"):
        client = requests.Session()
        retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        client.mount("https://", HTTPAdapter(max_retries=retry))
        THREAD_LOCAL.session = client
    return THREAD_LOCAL.session


def metadata(repo):
    response = session().get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60)
    response.raise_for_status()
    return response.json()


def read_small(repo, revision, name, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    response = session().get(
        f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true", timeout=60,
    )
    response.raise_for_status()
    target.write_bytes(response.content)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_range(job):
    repo, revision, name, target, start, end = job
    part = target.with_name(f"{target.name}.part-{start}")
    size = end - start + 1
    if not part.is_file() or part.stat().st_size != size:
        for attempt in range(5):
            try:
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}?dmd_part={start}"
                with session().get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/"
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                        raise RuntimeError(f"Unexpected byte range for {name}: {response.status_code}")
                    with part.open("wb") as output:
                        for block in response.iter_content(256 * 1024):
                            output.write(block)
                if part.stat().st_size != size:
                    raise RuntimeError(f"Incomplete range for {name}")
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))
    with LOCK:
        PROGRESS["bytes"] += size
        PROGRESS["chunks"] += 1


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = ROOT.parent / "source-manifests"
    manifests.mkdir(exist_ok=True)
    official = metadata("stabilityai/stable-diffusion-3-medium-diffusers")
    original = {item["rfilename"]: item for item in official["siblings"]}
    sources = [
        ("frankjoshua/stable-diffusion-3-medium-diffusers", "sd3"),
        ("yuvalkirstain/PickScore_v1", "pickscore"),
        ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "processor"),
    ]
    downloads, large, jobs = [], [], []
    for repo, role in sources:
        info = metadata(repo)
        (manifests / (repo.replace("/", "--") + ".json")).write_text(json.dumps(info, indent=2))
        revision = info["sha"]
        if role == "sd3":
            directory = MODEL_ROOT / "sd3-medium"
        else:
            cache_root = CACHE / ("models--" + repo.replace("/", "--"))
            directory = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True, exist_ok=True)
            (cache_root / "refs/main").write_text(revision)
        downloads.append(dict(repo=repo, revision=revision, directory=str(directory)))
        for item in info["siblings"]:
            name = item["rfilename"]
            if role == "processor":
                selected = name in {"preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                                    "special_tokens_map.json", "vocab.json", "merges.txt"}
            elif role == "pickscore":
                selected = name in {"config.json", "model.safetensors"}
            else:
                selected = name.endswith((".json", ".txt", ".model", ".safetensors")) or name in {"README.md", "LICENSE", "LICENSE.md"}
            if not selected:
                continue
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            lfs = item.get("lfs")
            if lfs and name.endswith(".safetensors"):
                if role == "sd3":
                    assert lfs["sha256"] == original[name]["lfs"]["sha256"], name
                if target.is_file() and target.stat().st_size == lfs["size"] and sha256(target) == lfs["sha256"]:
                    print("ALREADY_VERIFIED", name, flush=True)
                    continue
                large.append((target, lfs))
                for start in range(0, lfs["size"], CHUNK):
                    end = min(start + CHUNK, lfs["size"]) - 1
                    PROGRESS["total"] += end - start + 1
                    # All files share a worker pool, so small and large components progress together.
                    jobs.append((repo, revision, name, target, start, end))
            else:
                read_small(repo, revision, name, target)
    (manifests / "download_sources.json").write_text(json.dumps(downloads, indent=2))
    started = time.monotonic()
    print("DOWNLOAD_START", dict(PROGRESS), "workers", WORKERS, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        pending = {executor.submit(download_range, job) for job in jobs}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=20,
                                                   return_when=concurrent.futures.FIRST_EXCEPTION)
            for result in done:
                result.result()
            elapsed = time.monotonic() - started
            with LOCK:
                print("DOWNLOAD_PROGRESS", dict(PROGRESS), "seconds", round(elapsed),
                      "MiB/s", round(PROGRESS["bytes"] / max(elapsed, 1) / 2**20, 2), flush=True)
    for target, lfs in large:
        digest = hashlib.sha256()
        temporary = target.with_name(target.name + ".assembling")
        with temporary.open("wb") as output:
            for start in range(0, lfs["size"], CHUNK):
                part = target.with_name(f"{target.name}.part-{start}")
                block = part.read_bytes()
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != lfs["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {target}")
        temporary.replace(target)
        for start in range(0, lfs["size"], CHUNK):
            target.with_name(f"{target.name}.part-{start}").unlink()
        print("VERIFIED", target, lfs["sha256"], flush=True)
    link = ROOT / "models/sd3-medium"
    link.parent.mkdir(exist_ok=True)
    if not link.exists() and not link.is_symlink():
        link.symlink_to(MODEL_ROOT / "sd3-medium", target_is_directory=True)
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
