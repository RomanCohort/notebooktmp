"""RNAQwen-preview pack: save / load an already-quantized model.

Why this exists
---------------
Training and eval stream the bf16 base (26 safetensors shards, 67 GiB) layer by
layer, quantize the MoE experts to nf4 on the fly, then drop the shard
mappings.  Peak system commit is 90-100 GB on this box (64 GB RAM + 64 GB
pagefile -> 127.6 GB commit limit).  That is the wall this project kept hitting:
`OSError: [Errno 1455]` on one path, a bare `0xC0000005` access violation on
another (PLAN_v2.md section 18).

A preview pack stores the already-quantized weights instead: ~23 GB of
safetensors, read once, no bf16 shard mappings, no on-the-fly quantization.

Layout written by save_pack()
-----------------------------
    <out>/weights/model-000NN-of-000MM.safetensors
    <out>/weights/index.json      key -> shard, per-shard sha256 + byte count

Format notes (measured on this box: bnb 0.50.2, torch 2.12+rocm7.13, transformers 5.9)
-------------------------------------------------------------------------------------
`Linear4bit.state_dict()` emits six *tensor* keys per module:

    <mod>.weight
    <mod>.weight.absmax
    <mod>.weight.quant_map
    <mod>.weight.nested_absmax
    <mod>.weight.nested_quant_map
    <mod>.weight.quant_state.bitsandbytes__nf4

Everything needed to rebuild the parameter is a tensor, so a plain safetensors
round trip is lossless.  The rebuild must go through
`Params4bit.from_prequantized`: `Linear4bit.load_state_dict` rejects these keys
("Unexpected key(s) in state_dict").  Direct probe on a 256x256 nf4 layer:
max|diff| == 0.0 against the original module.

All six keys of one module must stay in the same shard (group_keys enforces
this), otherwise a module cannot be rebuilt from one file.

The adapter is NOT merged into the weights
-----------------------------------------
`load_pack(adapter=True)` reads `sos.json` + the adapter file it names, rebuilds
the SOS layers with replace_linear_with_sos and loads the adapter tensors
exactly.  Measured on the D_qwen36_sos pack: folding the rank-8 delta into the
310 bf16 attention / shared_expert weights moves top-1 logits on 4 of 128 probe
positions (max|dlogit| 2.32, 9% of max logit) -- bf16 has 7 mantissa bits, so
the merged weight is a different function, not a rounding of the same one.
Since the whole point of the pack is to be the trained model, the delta stays
out and the loader rebuilds it.  `scripts/97_export_preview.py --merge` restores
the lossy merged form if a plain model is really wanted.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import torch

WEIGHTS_DIR = "weights"
INDEX_NAME = "index.json"
FORMAT = "rnaqwen-preview/1"

_WEIGHT = ".weight"
_STAT = ".weight."


def _module_of(key: str) -> str | None:
    """Module path of a bnb quantized-stat key, else None."""
    if _STAT in key:
        return key.split(_STAT, 1)[0]
    return None


def group_keys(keys: Iterable[str]) -> list[list[str]]:
    """Group keys so one quantized module never splits across shards."""
    keys = list(keys)
    qmods = {_module_of(k) for k in keys if _module_of(k) is not None}
    order: list[str] = []
    bucket: dict[str, list[str]] = {}
    for k in keys:
        m = _module_of(k)
        if m is not None:
            gid = m
        elif k.endswith(_WEIGHT) and k[: -len(_WEIGHT)] in qmods:
            gid = k[: -len(_WEIGHT)]
        else:
            gid = k
        if gid not in bucket:
            bucket[gid] = []
            order.append(gid)
        bucket[gid].append(k)
    return [bucket[g] for g in order]


def split_shard(raw: dict) -> tuple[dict, dict]:
    """Split one shard's tensors into (plain tensors, quantized groups)."""
    groups: dict[str, dict] = {}
    plain: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        m = _module_of(k)
        if m is None:
            plain[k] = v
        else:
            groups.setdefault(m, {})[k.split(_STAT, 1)[1]] = v
    for mod, g in groups.items():
        w = plain.pop(mod + _WEIGHT, None)
        if w is None:
            raise KeyError(f"quantized group {mod} has stats but no own .weight")
        g["weight"] = w
    return plain, groups


def _sha256(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(block), b""):
            h.update(blk)
    return h.hexdigest()


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def save_pack(llm: torch.nn.Module, out, shard_gb: float = 3.0,
              log: Callable = print) -> dict:
    """Write the (already quantized) state dict of `llm` as sharded safetensors."""
    from safetensors.torch import save_file

    out = Path(out)
    wdir = out / WEIGHTS_DIR
    wdir.mkdir(parents=True, exist_ok=True)
    for old in wdir.glob("*.safetensors"):
        old.unlink()

    sd = llm.state_dict()
    plan: list[list[str]] = []
    cur: list[str] = []
    cur_gb = 0.0
    for g in group_keys(sd.keys()):
        gb = sum(_nbytes(sd[k]) for k in g) / 2 ** 30
        if cur and cur_gb + gb > shard_gb:
            plan.append(cur)
            cur, cur_gb = [], 0.0
        cur.extend(g)
        cur_gb += gb
    if cur:
        plan.append(cur)

    n = len(plan)
    shards: list[dict] = []
    key_to_shard: dict[str, str] = {}
    t0 = time.time()
    for i, keys in enumerate(plan, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        path = wdir / name
        save_file({k: sd[k].detach().to("cpu").contiguous() for k in keys},
                  str(path), metadata={"format": "pt"})
        total = sum(_nbytes(sd[k]) for k in keys)
        shards.append({"name": name, "keys": len(keys), "bytes": total,
                       "sha256": _sha256(path)})
        for k in keys:
            key_to_shard[k] = name
        log(f"  [pack] {name}  {len(keys):>5} keys  {total / 2 ** 30:>5.2f} GiB"
            f"  {time.time() - t0:>6.1f}s")
    del sd
    gc.collect()

    index = {"format": FORMAT, "shard_gb": shard_gb, "shards": shards,
             "key_to_shard": key_to_shard,
             "total_bytes": sum(s["bytes"] for s in shards),
             "total_keys": len(key_to_shard)}
    (wdir / INDEX_NAME).write_text(json.dumps(index, indent=1), encoding="utf-8")
    log(f"  [pack] {n} shards  {index['total_bytes'] / 2 ** 30:.2f} GiB  "
        f"{index['total_keys']} keys  {time.time() - t0:.1f}s")
    return index


def _resolve(root: torch.nn.Module, path: str):
    """Walk a dotted module path; numeric parts index ModuleList/Sequential."""
    parts = path.split(".")
    mod = root
    for p in parts[:-1]:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod, parts[-1]


def _prequantized(lin: torch.nn.Module, group: dict, dev) -> torch.nn.Module:
    """nn.Linear -> Linear4bit carrying the saved nf4 tensors (no requantize)."""
    from bitsandbytes.nn import Linear4bit, Params4bit

    data = group.pop("weight")
    new = Linear4bit(lin.in_features, lin.out_features,
                     bias=lin.bias is not None,
                     compute_dtype=torch.bfloat16, compress_statistics=True,
                     quant_type="nf4")
    new.weight = Params4bit.from_prequantized(
        data=data, quantized_stats=group, requires_grad=False,
        device=dev, module=new)
    if lin.bias is not None and lin.bias.device.type != "meta":
        new.bias = torch.nn.Parameter(lin.bias.detach().to(dev).clone())
    return new


def load_pack(out, dev=None, log: Callable = print, expect_sha: bool = False,
              adapter: bool = True):
    """Rebuild the quantized VLM from a pack written by save_pack()."""
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    from safetensors import safe_open
    from transformers import AutoConfig

    from .qwen36 import (fix_meta_buffers, patch_conv1d, patch_moe_experts,
                         trim_working_set)

    out = Path(out)
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(str(out))
    patch_moe_experts(M)
    patch_conv1d(M)
    with torch.device("meta"):
        model = M.Qwen3_5MoeForConditionalGeneration(cfg)
    text = model.model.language_model
    fx = fix_meta_buffers(model, cfg.text_config, dev)
    if fx:
        log(f"  [load] meta buffer 已重建 {fx}")
    if hasattr(model.model, "visual"):
        del model.model.visual
    if hasattr(model, "mtp"):
        del model.mtp
    gc.collect()

    index = json.loads((out / WEIGHTS_DIR / INDEX_NAME).read_text(encoding="utf-8"))
    by_shard: dict[str, list[str]] = {}
    for k, s in index["key_to_shard"].items():
        by_shard.setdefault(s, []).append(k)

    t0 = time.time()
    done = 0
    nq_all = 0
    for s in index["shards"]:
        path = out / WEIGHTS_DIR / s["name"]
        if expect_sha and _sha256(path) != s["sha256"]:
            raise RuntimeError(f"分片校验和不符：{s['name']}")
        keys = by_shard.get(s["name"], [])
        with safe_open(str(path), framework="pt", device="cpu") as f:
            raw = {k: f.get_tensor(k) for k in keys}
        plain, groups = split_shard(raw)
        model.load_state_dict(plain, strict=False, assign=True)
        for mod_path, g in groups.items():
            parent, name = _resolve(model, mod_path)
            setattr(parent, name, _prequantized(getattr(parent, name), g, dev))
        nq_all += len(groups)
        done += len(keys)
        del raw, plain, groups
        gc.collect()
        torch.cuda.empty_cache()
        trim_working_set()
        log(f"  [load] {s['name']}  {len(keys):>5} keys  量化 {nq_all:>6}  "
            f"{time.time() - t0:>6.1f}s")

    for i in range(len(text.layers)):
        text.layers[i].to(dev)
    for m in (getattr(text, "embed_tokens", None), getattr(text, "norm", None),
              getattr(text, "rotary_emb", None), getattr(model, "lm_head", None)):
        if m is not None:
            m.to(dev)
    gc.collect()
    torch.cuda.empty_cache()

    sosp = out / "sos.json"
    if adapter and sosp.exists():
        from .models.sos_lora import replace_linear_with_sos

        sc = json.loads(sosp.read_text(encoding="utf-8"))
        rep = replace_linear_with_sos(model, tuple(sc["targets"]), sc["r_tot"],
                                      sc["k"], sc["alpha"],
                                      dropout=sc.get("dropout", 0.0),
                                      gamma_max=sc["gamma_max"])
        sd = torch.load(out / sc["file"], map_location="cpu",
                        weights_only=False)
        miss = model.load_state_dict(sd, strict=False)
        want = {n for n, p in model.named_parameters() if p.requires_grad}
        extra = list(getattr(miss, "unexpected_keys", None) or [])
        if extra or set(sd) != want:
            raise RuntimeError(
                f"SOS 适配器与底座对不上：unexpected {extra[:3]}  "
                f"缺失 {sorted(want - set(sd))[:3]}")
        model.eval()
        del sd
        gc.collect()
        log(f"  [load] SOS 适配器 -> 重建 {len(rep)} 层（未合并）  "
            f"K={sc['k']} r_tot={sc['r_tot']} scales_alpha={sc['alpha']}")

    n_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if n_meta:
        raise RuntimeError(f"{len(n_meta)} 个参数仍在 meta：{n_meta[:4]}")
    if done != index["total_keys"]:
        raise RuntimeError(f"装载 {done} 个 key，索引里有 {index['total_keys']} 个")
    log(f"  [load] 完成  {nq_all} 个预量化模块  {done} keys  "
        f"{time.time() - t0:.1f}s")
    return model
