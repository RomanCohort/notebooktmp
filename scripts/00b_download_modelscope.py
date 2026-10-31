"""从 ModelScope 下载模型权重（多路并发 + 断点续传）。

为什么走 ModelScope
-------------------
实测本机到 HuggingFace CDN（us.aws.cdn.hf.co）：

    hf_direct   0.06 MB/s
    hf_mirror   0.06 MB/s        <- 也是回源到 hf.co，没用

    而 ModelScope:
    modelscope  11.21 MB/s       <- 快 180 倍

Qwen3-8B 是 16.4 GB —— 走 HF 要 45 小时，走 ModelScope 25 分钟。
所以权重一律从 ModelScope 拉；拉不到的（小众学术权重）再回 HF 慢慢磨。

用法：
    python scripts/00b_download_modelscope.py --repo Qwen/Qwen3-8B --dest models/Qwen3-8B --workers 5
    python scripts/00b_download_modelscope.py --list --repo Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MS_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
MS_RESOLVE = "https://modelscope.cn/models/{repo}/resolve/{rev}/{path}"

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def list_files(repo: str, revision: str = "master") -> list[dict]:
    url = f"{MS_API.format(repo=repo)}?Revision={urllib.parse.quote(revision)}&Recursive=True"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "rnaqwen/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    data = payload.get("Data") or {}
    files = data.get("Files") or []
    out = []
    for f in files:
        if f.get("Type") == "tree" or f.get("Size") is None:
            continue
        out.append({"path": f["Path"], "size": int(f["Size"]), "sha256": f.get("Sha256") or f.get("Sha256Hash")})
    return out


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download_one(repo: str, revision: str, path: str, size: int, dest: Path, retries: int = 8) -> tuple[str, int, float]:
    """下载单个文件，支持断点续传 + **卡死看门狗**。

    踩过的坑：用 `urllib.request.urlopen(timeout=120)` 时，连接一旦半死不活
    （CDN 返回 200 但不再吐字节），read 会一直挂着 —— 实测 5 个 shard 全部
    停在 0.00 MB/s，静默不动。
    改用 `requests` 的元组超时 `(连接超时, 读超时)`：读超时一到就抛异常，
    然后带 `Range` 续传重试。这是唯一能自动从"半死连接"里爬出来的办法。
    """
    import requests

    # 先写 `.part`，完整拿到再原子改名。
    # 教训（真踩过）：直接写目标文件时，如果两个进程同时在写同一个文件，
    # 或者两轮续传拼了不同 CDN 节点的字节，结果就是**大小完全对得上、
    # 内容却是坏的**静默损坏 —— safetensors 能加载，模型只会吐乱码。
    # Qwen3-8B 的 1/2/3/4 号 shard 就是这么坏的（sha256 全对不上，size 全对）。
    final = dest / path
    target = dest / (path + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = MS_RESOLVE.format(repo=repo, rev=revision, path=urllib.parse.quote(path))
    t0 = time.time()

    if final.exists() and final.stat().st_size == size and size > 0:
        return path, size, 0.0

    stall_retries = 0
    for attempt in range(retries):
        have = target.stat().st_size if target.exists() else 0
        if have > size:  # 上次写坏了，重来
            target.unlink()
            have = 0
        headers = {"User-Agent": "rnaqwen/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            # 关键：**读超时要短**（10s）。CDN 会「滴水」——连接不报错但几乎不吐数据，
            # 长读超时会让进程一直挂着（实测 5 个 shard 全停在 0.00 MB/s 且不抛异常）。
            with requests.get(url, headers=headers, stream=True, timeout=(15, 10), allow_redirects=True) as resp:
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                mode = "ab" if (have and resp.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                slow_since, slow_bytes = time.time(), have
                with target.open(mode) as fh:
                    for chunk in resp.iter_content(256 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        # 速率看门狗：45s 内增量不到 200KB 就当卡死，断开重连续传
                        if now - slow_since > 45:
                            rate = (have - slow_bytes) / (now - slow_since)
                            if rate < 200_000:
                                raise RuntimeError(f"stalled: {rate/1000:.1f} KB/s over {now-slow_since:.0f}s")
                            slow_since, slow_bytes = now, have
            if target.stat().st_size == size:
                os.replace(target, final)   # 原子改名：只有完整文件才会出现在最终路径
                return path, size, time.time() - t0
            time.sleep(1.0 + attempt)
        except Exception as exc:  # noqa: BLE001
            stall_retries += 1
            got = target.stat().st_size if target.exists() else 0
            log(f"    [retry {attempt+1}/{retries}] {path}: {type(exc).__name__} {str(exc)[:70]} "
                f"({human(got)}/{human(size)})")
            time.sleep(min(1.5 * stall_retries, 20.0))
    return path, target.stat().st_size if target.exists() else 0, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--revision", default="master")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--include", default=None, help="只下匹配的子串（逗号分隔）")
    ap.add_argument("--exclude", default=None, help="跳过匹配的子串（逗号分隔）")
    args = ap.parse_args()

    dest = args.dest or Path("models") / args.repo.split("/")[-1]
    files = list_files(args.repo, args.revision)
    if args.include:
        pats = [p.strip() for p in args.include.split(",") if p.strip()]
        files = [f for f in files if any(p in f["path"] for p in pats)]
    if args.exclude:
        pats = [p.strip() for p in args.exclude.split(",") if p.strip()]
        files = [f for f in files if not any(p in f["path"] for p in pats)]

    total = sum(f["size"] for f in files)
    log(f"[ms] {args.repo}@{args.revision} -> {dest}")
    log(f"[ms] {len(files)} 个文件，共 {human(total)}")
    if args.list:
        for f in files:
            log(f"   {human(f['size']):>10s}  {f['path']}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, args.repo, args.revision, f["path"], f["size"], dest): f for f in files}
        for fut in as_completed(futs):
            path, got, secs = fut.result()
            done += 1
            elapsed = max(time.time() - t0, 1e-6)
            mbps = got / secs / 1e6 if secs > 0 else 0
            log(f"  [{done}/{len(files)}] {human(got):>9s} {mbps:6.2f} MB/s  {path}  (累计 {human(total*(done/len(files)))}/{human(total)})")

    have = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    log(f"[ms] done: {human(have)} / {human(total)}  用时 {(time.time()-t0)/60:.1f} min -> {dest}")
    return 0 if have >= total else 1


if __name__ == "__main__":
    sys.exit(main())
