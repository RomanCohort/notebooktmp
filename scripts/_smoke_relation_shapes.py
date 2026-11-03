"""Shape/gradient smoke test for the upgraded relation model. No 35B load.

Builds a tiny random-weight Qwen3ForCausalLM plus a fake ERNIE encoder, runs the
mixed (target + region) batch through RelationModel.forward and predict, and
asserts every new path actually receives gradient.

    python scripts/_smoke_relation_shapes.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401  shim

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

spec = importlib.util.spec_from_file_location(
    "r52", ROOT / "scripts/52_routeR_relation.py")
r52 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r52)

ENC_DIM = 768
LLM_DIM = 64
N_QUERY = 4
ACC_DIM = 50


class FakeTok:
    """Only what make_batch touches: enc, eos, pad_id, split."""
    pad_id = 0
    eos = "<eos>"

    def enc(self, s):
        # Every answer text must be exactly ONE token, exactly like the real
        # tokenizer (the trainer asserts this).  Question text may be any length.
        single = {"是": 900, "否": 901,
                  "3'UTR": 700, "5'UTR": 701, "CDS": 702, "Intron": 703,
                  "<eos>": 2}
        if s in single:
            return [single[s]]
        # answer + eos: make_batch calls tok.enc(a + tok.eos)
        for a, tid in (("3'UTR", 700), ("5'UTR", 701), ("CDS", 702),
                       ("Intron", 703), ("是", 900), ("否", 901)):
            if s == a + self.eos:
                return [tid, 2]
        return [100 + (sum(map(ord, s)) % 50) for _ in range(max(1, len(s) % 7 + 1))]

    def split(self, q):
        return [1, 2, 3], [4] * (36 if "靶向" in q else 41)


class FakeEncoder(torch.nn.Module):
    hidden_size = ENC_DIM

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(64, ENC_DIM)

    def forward(self, seqs):
        L = max(len(s) for s in seqs) + 2      # BOS + EOS
        B = len(seqs)
        dev = self.emb.weight.device
        h = self.emb(torch.zeros(B, L, dtype=torch.long, device=dev))
        mask = torch.ones(B, L, dtype=torch.long, device=dev)
        return types.SimpleNamespace(hidden=h, mask=mask)


class FakeBase(torch.nn.Module):
    """Mimics the VLM wrapper: .model.language_model + .lm_head."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = Qwen3ForCausalLM(
            Qwen3Config(hidden_size=LLM_DIM, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2,
                        intermediate_size=128, vocab_size=1024)).model
        self.lm_head = torch.nn.Linear(LLM_DIM, 1024, bias=False)


def build_batch(n_target=2, n_region=2):
    q_t = "这条 miRNA 与这段 mRNA 窗口之间是否存在经实验验证的靶向关系？只回答「是」或「否」。"
    q_r = "这条 miRNA 的结合位点位于哪个区域？只回答「3'UTR」「5'UTR」「CDS」「Intron」之一。"
    rows = []
    for i in range(n_target):
        rows.append(dict(mirna="ACGTACGTACGTACGTACGTAC", mrna_window="A" * 50,
                         question=q_t, answer="是" if i == 0 else "否",
                         label=1 if i == 0 else 0, kind="mirna_target"))
    for i, ans in enumerate(["CDS", "Intron"][:n_region]):
        rows.append(dict(mirna="TTTTGGGGCCCCAAAATTTTGG", mrna_window="C" * 50,
                         question=q_r, answer=ans, label=1, kind="mirna_region"))
    acc = {r["mrna_window"]: [0.5] * ACC_DIM for r in rows}
    return rows, acc


def main() -> int:
    tok = FakeTok()
    rows, acc = build_batch()
    b = r52.make_batch(rows, tok, n_soft=2 * N_QUERY, acc=acc, acc_dim=ACC_DIM)

    # ---- batch invariants ----
    assert b["prefix_ids"].shape[0] == len(rows), b["prefix_ids"].shape
    base = b["prefix_ids"].shape[1] + 2 * N_QUERY
    assert torch.equal(b["is_region"].cpu(),
                       torch.tensor([False, False, True, True]))
    assert set(b["region_ids"][b["is_region"]].tolist()) == {2, 3}
    assert b["acc"].shape == (len(rows), ACC_DIM)
    print(f"[ok] batch prefix {tuple(b['prefix_ids'].shape)} pred_pos "
          f"{b['pred_pos'].tolist()} base {base}")

    # ---- forward / backward ----
    dev = r52.DEV
    enc = FakeEncoder().to(dev)
    base_m = FakeBase().to(dev)
    # dtype 默认 bf16，和真实训练一致 —— 用 fp32 假模型测不出
    # "region_head 是 fp32 而 sel 是 bf16" 这类错误。
    model = r52.RelationModel(base_m, enc, llm_dim=LLM_DIM, enc_dim=ENC_DIM,
                              n_query=N_QUERY, acc_dim=ACC_DIM,
                              dtype=torch.bfloat16)
    model.rs_mi.to(dev); model.rs_mr.to(dev); model.region_head.to(dev)

    out = model.forward(b)
    assert set(out) == {"target", "region"}, out.keys()
    assert torch.isfinite(out["target"]) and torch.isfinite(out["region"])
    loss = out["target"] + 0.1 * out["region"]
    loss.backward()

    def g(mod, name):
        p = dict(mod.named_parameters()).get(name)
        return None if p is None or p.grad is None else float(p.grad.abs().sum())

    assert g(model.rs_mr, "feat_proj.weight") and g(model.rs_mr, "feat_proj.weight") > 0, \
        "rs_mr.feat_proj 没有梯度"
    assert g(model.region_head, "weight") and g(model.region_head, "weight") > 0, \
        "region_head 没有梯度"
    assert model.rs_mi.feat_proj is None, "rs_mi 不该有 feat_proj"
    print(f"[ok] forward target {float(out['target']):.4f} "
          f"region {float(out['region']):.4f}; "
          f"feat_proj grad {g(model.rs_mr, 'feat_proj.weight'):.2e}, "
          f"region_head grad {g(model.region_head, 'weight'):.2e}")

    # ---- predict must not touch region_head ----
    model.eval()
    pb = r52.make_batch(rows, tok, with_answer=False, n_soft=2 * N_QUERY,
                        acc=acc, acc_dim=ACC_DIM)
    with torch.no_grad():
        lg = model.predict(pb)
    assert lg.shape == (len(rows), 1024), lg.shape
    print(f"[ok] predict {tuple(lg.shape)}")

    # ---- guard: wrong accessibility width must raise ----
    bad = dict(b)
    bad["acc"] = torch.zeros(len(rows), ACC_DIM - 1)
    try:
        model.eval()
        with torch.no_grad():
            model.forward(bad)
    except RuntimeError as e:
        print(f"[ok] 长度守卫触发：{str(e)[:60]}")
    else:
        raise AssertionError("可及性宽度不对却没有报错")

    print("\n[pass] all smoke checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
