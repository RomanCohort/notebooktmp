# RNAQwen private-server runbook

This deployment trains the 35B int4 model with SOS-LoRA, the v3 dataset, and the
new `fsq_stream` codebook. The FSQ encoder is frozen because the data path uses
precomputed uint16 code IDs.

## 1. Host requirements

- Linux x86_64 with NVIDIA driver and `nvidia-smi`
- GPU with at least 30 GiB visible memory; A100 40GB is preferred
- 64-128 GiB RAM (128 GiB preferred)
- At least 100 GiB free for training; 170 GiB free if GGUF export is done here
- CUDA-compatible PyTorch installed before running the deployment
- `git`, `curl`, `flock`, `tmux`, `sha256sum`

Check:

```bash
nvidia-smi
df -h /opt
free -h
python3.13 --version
```

## 2. Install once

```bash
sudo useradd --system --create-home --shell /bin/bash rnaqwen
sudo mkdir -p /opt/RNAQwen /etc/rnaqwen /opt/RNAQwen/backups
sudo chown -R rnaqwen:rnaqwen /opt/RNAQwen

sudo -u rnaqwen git clone https://github.com/RomanCohort/notebooktmp.git /opt/RNAQwen
sudo -u rnaqwen python3.13 -m venv /opt/RNAQwen/.venv
sudo -u rnaqwen /opt/RNAQwen/.venv/bin/pip install --upgrade pip
```

Install the PyTorch wheel matching the host CUDA driver, for example:

```bash
sudo -u rnaqwen /opt/RNAQwen/.venv/bin/pip install \
  torch --index-url https://download.pytorch.org/whl/cu128
sudo -u rnaqwen /opt/RNAQwen/.venv/bin/pip install -r /opt/RNAQwen/requirements.txt
# CUDA only: enables the FLA/causal-conv1d fast path for linear attention.
sudo -u rnaqwen /opt/RNAQwen/.venv/bin/pip install \
  'flash-linear-attention[cuda]==0.6.0'
# Required: install causal-conv1d only from its matching official release wheel.
# Do not run plain `pip install causal-conv1d`; PyPI is source-only and will compile.
```

Verify:

```bash
sudo -u rnaqwen /opt/RNAQwen/.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

## 3. Configure

```bash
sudo cp /opt/RNAQwen/deploy/train.env.example /etc/rnaqwen/train.env
sudo chown root:rnaqwen /etc/rnaqwen/train.env
sudo chmod 640 /etc/rnaqwen/train.env
sudo -u rnaqwen chmod +x /opt/RNAQwen/deploy/*.sh
```

The first run downloads the 663 MB dataset, base model, and ERNIE-RNA. It then
builds the compact code cache before training.

## 4. Run with tmux first

Keep the server powered on. The SSH client and local computer may disconnect.

```bash
sudo -u rnaqwen env RNAQ_ENV_FILE=/etc/rnaqwen/train.env \
  /opt/RNAQwen/deploy/start_tmux.sh
tmux attach -t rnaqwen
```

Detach with `Ctrl+b`, then `d`.

Health checks:

```bash
tmux ls
tail -f /opt/RNAQwen/logs/deploy-v3_full.log
nvidia-smi
find /opt/RNAQwen/runs/e2e_v3_full/ckpt -name 'step_*.pt' -printf '%f\n' | sort | tail
ls -lh /opt/RNAQwen/backups
```

Checkpoints are saved every 100 steps or 10 minutes. The newest local backup is
`/opt/RNAQwen/backups/v3_full-latest.pt`.

Stop after a checkpoint:

```bash
tmux send-keys -t rnaqwen C-c
```

Resume:

```bash
sudo -u rnaqwen env RNAQ_ENV_FILE=/etc/rnaqwen/train.env \
  /opt/RNAQwen/deploy/start_tmux.sh
```

`run_train.sh` discovers the newest `step_*.pt` and passes it to `--resume`.

## 5. Optional systemd service

Use this only after the tmux run has passed smoke/model loading checks.

```bash
sudo cp /opt/RNAQwen/deploy/rnaqwen-train.service.example \
  /etc/systemd/system/rnaqwen-train.service
sudo systemctl daemon-reload
sudo systemctl enable --now rnaqwen-train
sudo systemctl status rnaqwen-train
journalctl -u rnaqwen-train -f
```

Stop and resume:

```bash
sudo systemctl stop rnaqwen-train
sudo systemctl start rnaqwen-train
```

Do not run tmux and systemd at the same time. The `flock` guard prevents two
trainers but one will simply exit.

## 6. Recovery and rollback

If the trainer exits:

```bash
journalctl -u rnaqwen-train -n 200 --no-pager
tail -n 200 /opt/RNAQwen/logs/deploy-v3_full.log
```

If the latest checkpoint is damaged, list snapshots:

```bash
find /opt/RNAQwen/runs/e2e_v3_full/ckpt -name 'step_*.pt' -ls
```

Restore the backup:

```bash
cp /opt/RNAQwen/backups/v3_full-latest.pt \
  /opt/RNAQwen/runs/e2e_v3_full/ckpt/step_restore.pt
```

Then restart with `--resume` pointing at the restored file if needed. Do not
delete the old checkpoints until the restored run has produced a new valid
checkpoint.

## 7. GGUF export is separate

Do not run GGUF export by default on the training filesystem. Q8_0 plus Q4_K_M
adds about 55 GiB. Use the full local RNAQwen workspace (which contains
`scripts/97_export_preview.py` and `scripts/98_export_gguf.py`) on a machine
with at least 170 GiB free:

```bash
python scripts/98_export_gguf.py --phase patch \
  --ckpt runs/e2e_v3_full/ckpt/final.pt \
  --fsq checkpoints/fsq_stream/tokenizer.pt
python scripts/98_export_gguf.py --phase convert
python scripts/98_export_gguf.py --phase quantize --qtype Q4_K_M
python scripts/98_export_gguf.py --phase assets
python scripts/98_export_gguf.py --phase verify
```

## 8. Expected startup output

Healthy startup includes:

```text
gpu=... memory=...MiB
dataset ...
[fsq] latent=12 levels=2 vocab=4096
[codes] memmap ok ... vocab=4096
[base] Qwen3.6-35B-A3B mode=int4 adapter=sos
[sos] ... 替换 190 个 nn.Linear K=4 r_tot=8
[sos] 可训 8.13 M
```

If the log reports `adapter=lora`, the wrong flag was used. The intended flag is
`--adapter sos`.
